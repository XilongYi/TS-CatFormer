import torch.nn as nn
import torch
from src.model.common_layers.positionwise_feed_forward import PositionwiseFeedForward
import torch.nn.functional as F
import shelve
import os
import re


def init_embedding(layer):
    if isinstance(layer, nn.Embedding):
        if len(layer.weight) > 1:
            layer.weight.data[0].zero_()
    if isinstance(layer, nn.Linear) and layer.bias is not None:
        layer.bias.data.zero_()


def build_cat_mask_from_aa_dict(aa_dict_batch, asites_label_batch, N, device):
    B = len(aa_dict_batch)
    cat_mask_atom = torch.zeros((B, N), dtype=torch.bool, device=device)

    for b in range(B):
        aa_dict = aa_dict_batch[b]
        label = asites_label_batch[b]
        if label is None:
            cat_res = set()
        else:
            cat_res = set(int(x) for x in re.findall(r"\d+", str(label)))
        if not cat_res:
            continue

        for k, atom_ids in aa_dict.items():
            try:
                res_id = int(k.split("_", 1)[0])
            except Exception:
                continue

            if res_id in cat_res:
                for idx in atom_ids:
                    if 0 <= idx < N:
                        cat_mask_atom[b, idx] = True

    return cat_mask_atom

class VinaScoreHead(nn.Module):
    def __init__(self, hidden_dim, node_featurizer, dropout_rate=0.2, energy_output_folder='', 
                 edge_output_layer=True, if_energy_output=False, mode_name=None):
        super(VinaScoreHead, self).__init__()
        self.cef_mode = (mode_name == 'cef')
        self.num_atom_types = 29
        self.train_dist_cut_off = 4.0  # cut_off + 1.9 * 2 = real_distance, approximately 8.0 A
        self.node_featurizer = node_featurizer
        # trainable parameters
        self.pair_ln = nn.LayerNorm(hidden_dim)
        self.node_ln = nn.LayerNorm(hidden_dim)
        self.final_pair_ln = nn.LayerNorm(hidden_dim)
        # Gaussian Heads
        self.meanHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout_rate=dropout_rate)
        self.sigmaHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout_rate=dropout_rate)
        self.WeightHead = PositionwiseFeedForward(hidden_dim, hidden_dim * 2, d_out=4, dropout_rate=dropout_rate)
        # init
        self.init_param()
        # Vina Section
        self.vdw_table = self.make_vdw_table(node_featurizer)
        self.make_hydro_hbond_table()
        # Output Energy
        self.energy_output_folder = energy_output_folder
        print(f"# [Interformer] energy_output_fodler:{energy_output_folder}")
        # Ablation Study
        self.use_edge_output_layer = edge_output_layer
        self.if_energy_output = if_energy_output

    def init_param(self):
        self.apply(lambda module: init_embedding(module))

    def make_vdw_table(self, node_featurizer):
        edge_space = 29 ** 2
        table = torch.zeros([edge_space, 1])
        for i in range(28):
            for j in range(28):
                vdw_1 = node_featurizer.atom_type_data[i].xs_radius
                vdw_2 = node_featurizer.atom_type_data[j].xs_radius
                index = (1 + i) * 29 + (1 + j)
                table[index] = vdw_1 + vdw_2
        vdw_embedding = nn.Embedding(edge_space, 1).requires_grad_(False)
        vdw_embedding.weight = torch.nn.Parameter(table)
        vdw_embedding.weight.requires_grad = False
        return vdw_embedding

    def make_hydro_hbond_table(self):
        edge_space = 29 ** 2
        hydro_list = [3, 5, 18, 19, 20, 21]  # CH, ACH, F, CL, B, I
        acceptor_list = [9, 10, 13, 14]  # NDA, NA, ODA, OA
        donnor_list = [8, 9, 12, 13]  # XD, XDA, OD, ODA
        #
        hydro_table = torch.zeros([edge_space, 1])
        hbond_table = torch.zeros([edge_space, 1])
        for h_id in hydro_list:
            for h2_id in hydro_list:
                index = h_id * 29 + h2_id
                hydro_table[index] = 1.
        for a_id in acceptor_list:
            for d_id in donnor_list:
                index = a_id * 29 + d_id
                hbond_table[index] = 1.
                index = d_id * 29 + a_id
                hbond_table[index] = 1.
        # assign to embedding
        hydro_embedding = nn.Embedding(edge_space, 1).requires_grad_(False)
        hydro_embedding.weight = torch.nn.Parameter(hydro_table)
        hydro_embedding.weight.requires_grad = False
        self.hydro_table = hydro_embedding
        hbond_embedding = nn.Embedding(edge_space, 1).requires_grad_(False)
        hbond_embedding.weight = torch.nn.Parameter(hbond_table)
        hbond_embedding.weight.requires_grad = False
        self.hbond_table = hbond_embedding

    def atom_type2pair_type(self, x: torch.Tensor) -> torch.Tensor:
        # [b, n, 1]
        atoms_type_x = torch.where(x >= self.num_atom_types, x - self.num_atom_types,
                                   x)  # 29-th is the 0-idx in pocket-atoms-type
        pair_type = atoms_type_x[:, :, None, :] * self.num_atom_types + atoms_type_x[:, None, :]
        return pair_type

    def edge_output_layer(self, node_feats, pair_feats):
        pair_feats = pair_feats[:, 1:, 1:, :]  # [b, n, n, h]
        pair_feats = self.pair_ln(pair_feats)
        pair_feats = (pair_feats + pair_feats.transpose(1, 2)) * 0.5  # merge from L->P, and P->L
        # 2. node_feats=[b, n, d]
        node_feats = self.node_ln(node_feats[:, 1:])  # exclude vn node
        node_feats = torch.einsum('b i d, b j d-> b i j d', node_feats, node_feats)
        # Predict the closest pair
        final_pair_feats = node_feats + pair_feats
        final_pair_feats = self.final_pair_ln(final_pair_feats)
        return final_pair_feats

    def gaussian(self, d, mean, width):
        normal = torch.distributions.Normal(mean, width)
        logik = normal.log_prob(d.expand_as(normal.loc))
        return logik

    def mdn_loss(self, logPro, pi_soft, close_pairs_mask, y):
        close_pairs_mask = close_pairs_mask.squeeze(-1)  # [b, n, n]
        y_mask = (y >= 0.).view(-1)  # [b, 1, 1]
        b, n, _ = close_pairs_mask.shape
        # Cross-Entropy
        y_mask = y_mask.long().view(b, 1, 1)
        # default
        mix_loglik = torch.log(pi_soft) + logPro
        loglik = torch.logsumexp(mix_loglik, dim=-1)
        loglik = loglik * close_pairs_mask
        # neg_lik, it is hard to handle
        neg_lik = (pi_soft * torch.exp(logPro)).sum(dim=-1)
        neg_lik = torch.log((1. - neg_lik).clip(1e-9))
        neg_lik = neg_lik * close_pairs_mask
        # ce-loss
        ce_loss = -(y_mask * loglik + (1 - y_mask) * neg_lik)
        ce_mean_loss = ce_loss.mean()
        ### Log
        ce_pos_loss = -(y_mask * loglik).mean()
        return ce_mean_loss, ce_pos_loss
    
    def catalytic_energy_fraction_loss(
        self,
        score_pair, close_pairs_mask, pair_mask,
        ligand_mask, cat_mask, y_label, eps=1e-8,
        min_pairs=10,
    ):
        if ligand_mask.dim() == 3:
            ligand_mask = ligand_mask.squeeze(-1)
        if y_label.dim() == 2 and y_label.size(-1) == 1:
            y = y_label.squeeze(-1).float()
        else:
            y = y_label.float()

        lig_i = ligand_mask[:, :, None]          # [B,N,1]
        cat_j = cat_mask[:, None, :]             # [B,1,N]
        non_j = (~cat_mask)[:, None, :]          # [B,1,N]

        valid = close_pairs_mask & pair_mask     # [B,N,N]
        mask_cat    = valid & lig_i & cat_j
        mask_noncat = valid & lig_i & non_j

        mask_cat_f = mask_cat.float()
        mask_non_f = mask_noncat.float()

        num_cat = mask_cat_f.sum(dim=(1,2))
        num_non = mask_non_f.sum(dim=(1,2))
        valid_sample = (num_cat >= min_pairs) & (num_non >= min_pairs)

        # Mean aggregation.
        E_cat = (score_pair * mask_cat_f).sum(dim=(1,2)) / (num_cat + eps)
        E_non = (score_pair * mask_non_f).sum(dim=(1,2)) / (num_non + eps)

        f_cat = E_cat / (E_cat + E_non + eps)

        logits = (torch.log(E_cat + eps) - torch.log(E_non + eps)).clamp(-10, 10)

        if valid_sample.any():
            loss = F.binary_cross_entropy_with_logits(logits[valid_sample], y[valid_sample])
        else:
            loss = torch.zeros([], device=score_pair.device)

        return loss, f_cat.detach(), E_cat.detach(), E_non.detach()


    def GaussianScore(self, d, vdw_pair, pair_type, pair_mask, ligand_mask, pair_emb, batched_data):
        #
        close_pairs_mask = (d < self.train_dist_cut_off) & pair_mask
        hydro_pair = self.hydro_table(pair_type).bool()
        hbond_pair = self.hbond_table(pair_type).bool()

        # vdw-terms
        mean = F.elu(self.meanHead(pair_emb))
        sigma = F.elu(self.sigmaHead(pair_emb)) + 1. + 1e-5
        all_terms = self.gaussian(d, mean, sigma)

        vdw_term0 = all_terms[:, :, :, 0, None]
        vdw_term1 = all_terms[:, :, :, 1, None]
        hydro_term = all_terms[:, :, :, 2, None] * hydro_pair
        hbond_term = all_terms[:, :, :, 3, None] * hbond_pair

        # mixture weights
        pi = self.WeightHead(pair_emb)
        hydro_soft_mask = torch.where(hydro_pair, torch.tensor(0., device=d.device), torch.tensor(float('-inf'), device=d.device))
        hbond_soft_mask = torch.where(hbond_pair, torch.tensor(0., device=d.device), torch.tensor(float('-inf'), device=d.device))
        zero_soft_mask = torch.zeros_like(hydro_soft_mask)
        mask = torch.cat([zero_soft_mask, zero_soft_mask, hydro_soft_mask, hbond_soft_mask], dim=-1)

        pi_masked = pi + mask
        pi_soft = torch.softmax(pi_masked, dim=-1) + 1e-9

        loglik = torch.cat([vdw_term0, vdw_term1, hydro_term, hbond_term], dim=-1)
        # Handle pic50 fallback.
        if 'pic50' in batched_data and batched_data['pic50'] is not None:
            y_mdn = batched_data['pic50']
        else:
            # fallback: all-positive labels, shape [B, 1]
            B = loglik.size(0)
            y_mdn = torch.ones((B, 1), device=loglik.device, dtype=loglik.dtype)
                

        # Loss (batch-wise)
        mean_mdn_loss, mdn_pos_loss = self.mdn_loss(loglik, pi_soft, close_pairs_mask, y_mdn)

        # Save outputs.
        if self.if_energy_output:
            output_dir = f'{self.energy_output_folder}/gaussian_predict'
            os.makedirs(output_dir, exist_ok=True)

            B = d.shape[0]  # batch size
            for b in range(B):
                target = batched_data['target'][b]
                output_f = f'{output_dir}/{target}_G.db'

                with shelve.open(output_f) as db:
                    last_id = len(db) if len(db) else 0
                    print(f"[Interformer] Gaussian Score -> {target}-{last_id}")
                    data = {
                        'ligand_len': batched_data['ligand_len'][b].cpu().numpy(),
                        'pocket_len': batched_data['pocket_len'][b].cpu().numpy(),
                        'pi': pi_soft[b].detach().cpu().float().numpy(),
                        'mean': mean[b].detach().cpu().float().numpy(),
                        'sigma': sigma[b].detach().cpu().float().numpy(),
                        'hbond_pair': hbond_pair[b].detach().cpu().numpy(),
                        'hydro_pair': hydro_pair[b].detach().cpu().numpy(),
                        'd': d[b].detach().cpu().numpy(),
                        'vdw_pair': vdw_pair[b].detach().cpu().numpy(),
                    }
                    db[str(last_id)] = data
        if self.cef_mode:
            # score_pair = sum_k pi_k * exp(loglik_k)   => [B,N,N,1] -> squeeze
            score_pair = (pi_soft * torch.exp(loglik)).sum(dim=-1)      # [B, N, N]
            # Aggregate only close pairs.
            # Re-apply pair_mask for safety.

            # Build ligand, catalytic, and label masks.
            N = score_pair.size(1)
            cat_mask_atom = build_cat_mask_from_aa_dict(
                aa_dict_batch=batched_data['aa_dict'],
                asites_label_batch=batched_data['active_sites'],
                N=N,
                device=score_pair.device
            )
            y_label  = torch.tensor(
                batched_data['is_substrate'],
                dtype=torch.long,
                device=batched_data['x'].device
            ).view(-1, 1)                    # [B] or [B,1]
            pair_mask_ = batched_data['pair_mask'].squeeze(-1) if batched_data['pair_mask'].dim() == 4 else batched_data['pair_mask']  # [B,N,N]
            ligand_node_mask = ligand_mask.squeeze(-1)[:, :, 0]

            cef_loss, f_cat, E_cat, E_noncat = self.catalytic_energy_fraction_loss(
                score_pair=score_pair,
                close_pairs_mask=close_pairs_mask.squeeze(-1),
                pair_mask=pair_mask_,
                ligand_mask=ligand_node_mask,
                cat_mask=cat_mask_atom,
                y_label=y_label,
                eps=1e-8,
            )
            cef_stats = {'f_cat': f_cat, 'E_cat': E_cat, 'E_noncat': E_noncat}
        else:
            cef_loss = torch.tensor(0., device=d.device)

        return mean_mdn_loss, mdn_pos_loss, cef_loss

    def forward(self, node_feats, pair_feats, x_atom, batched_data):
        D, pair_mask, ligand_mask = batched_data['D'], batched_data['pair_mask'], batched_data[
            'ligand_mask']
        # Edge output Layer
        if self.use_edge_output_layer:
            pair_emb = self.edge_output_layer(node_feats, pair_feats)
        else:
            pair_emb = pair_feats[:, 1:, 1:, :]
        # Pair-Type
        
        pair_type = self.atom_type2pair_type(x_atom).squeeze(-1).long()
        vdw_pair = self.vdw_table(pair_type)  # [b, n, n, 1], (d < 5.)
        d = D - vdw_pair
        # EnergyScore
        gScore_loss, gScore_pos_loss, cef_loss = self.GaussianScore(d, vdw_pair, pair_type, pair_mask, ligand_mask, pair_emb,
                                                          batched_data)
    
        return 0., gScore_loss, gScore_pos_loss, cef_loss
