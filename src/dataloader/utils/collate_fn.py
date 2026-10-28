import torch
from src.utils.constant import *
from collections import defaultdict


def get_ligand_mask(D, ligand_len):
    ligand_mask = torch.zeros_like(D)
    for i, l in enumerate(ligand_len):
        ligand_mask[i, :l, :l] = 1.
    return ligand_mask.bool()


def uff_add_pocket_xyz(uff_xyz, gt_xyz, ligand_len):
    assert uff_xyz.shape == gt_xyz.shape
    for i, l_len in enumerate(ligand_len):
        uff_xyz[i, l_len:] = gt_xyz[i, l_len:]
    return uff_xyz


def unpack_dict(X):
    batch_dict = defaultdict(list)
    for row_dict in X:
        for key in row_dict:
            batch_dict[key].append(row_dict[key])
    return batch_dict


# Pad 2D features.
def pad_1D(x, padlen, pad_value=0.):
    xlen, d = x.size(0), x.size(1)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, d], dtype=x.dtype).fill_(float(pad_value))
        new_x[:xlen, :d] = x
        x = new_x
    return x.unsqueeze(0)


# Pad edge features.
def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def get_pair_and_pocket_mask(D, ligand_len, pocket_len, aa_len):
    """Return pair and pocket masks."""
    pair_mask = torch.zeros_like(D)
    pocket_mask = torch.zeros_like(D)

    for i, (l, p, a) in enumerate(zip(ligand_len, pocket_len, aa_len)):
        start_ligand = 0
        end_ligand = l

        start_pocket = end_ligand
        end_pocket = start_pocket + p

        start_aa = end_pocket
        end_aa = start_aa + a

        # Ligand-pocket edges.
        pair_mask[i, start_ligand:end_ligand, start_pocket:end_pocket] = 1.
        pair_mask[i, start_pocket:end_pocket, start_ligand:end_ligand] = 1.

        # Pocket-AA edges.
        pocket_mask[i, start_pocket:end_pocket, start_aa:end_aa] = 1.
        pocket_mask[i, start_aa:end_aa, start_pocket:end_pocket] = 1.

        # Pocket self-edges.
        pocket_mask[i, start_pocket:end_pocket, start_pocket:end_pocket] = 1.

        # AA self-edges.
        pocket_mask[i, start_aa:end_aa, start_aa:end_aa] = 1.

    # Clear padding and cast to bool.
    pair_mask = torch.where(D == -1., torch.tensor(0., device=pair_mask.device), pair_mask).bool()
    pocket_mask = torch.where(D == -1., torch.tensor(0., device=pocket_mask.device), pocket_mask).bool()
    return pair_mask, pocket_mask


def get_attn_bias(D, ndata, edata, dist_cut_off=True, aa_reaction_bias=True):
    B, max_N = ndata.size()[:2]
    attn_bias = torch.zeros([B, max_N + 1, max_N + 1], dtype=torch.float32).to(D)

    if dist_cut_off:
        too_far_away = ((D > attention_dist_cutoff) | (D == -1.))
    else:
        too_far_away = (D == -1.)

    attn_bias[:, 1:, 1:][too_far_away] = float("-inf")

    # Padding token mask.
    pad_matrix = (ndata[:, :, 0] == 0)
    attn_bias[:, 0, 1:][pad_matrix] = float("-inf")
    return attn_bias
def stack_if_not_none(x_list):
    """Stack tensors when available."""
    if x_list is None:
        return None
    if len(x_list) == 0:
        return None
    if x_list[0] is None:
        return None
    return torch.stack(x_list, dim=0)


def collate_pocket_graphs(graph_list):
    if graph_list is None:
        return None
    if len(graph_list) == 0:
        return None

    node_s_list = []
    node_v_list = []
    edge_s_list = []
    edge_v_list = []
    edge_index_list = []
    batch_list = []
    num_nodes = []
    x_list = []
    seq_list = []
    mask_list = []
    pdb_paths = []
    protein_ids = []
    batch_index = []
    valid_mask = []

    node_offset = 0
    for batch_idx, graph in enumerate(graph_list):
        graph_valid = (
            graph is not None
            and graph.get("is_valid", False)
            and graph.get("node_s", None) is not None
            and graph.get("node_v", None) is not None
            and graph.get("edge_index", None) is not None
            and graph.get("edge_s", None) is not None
            and graph.get("edge_v", None) is not None
        )
        valid_mask.append(graph_valid)
        pdb_paths.append(None if graph is None else graph.get("pdb_path", None))
        protein_ids.append(None if graph is None else graph.get("protein_id", None))
        if not graph_valid:
            continue

        cur_node_s = graph["node_s"]
        cur_node_v = graph["node_v"]
        cur_edge_s = graph["edge_s"]
        cur_edge_v = graph["edge_v"]
        cur_edge_index = graph["edge_index"]

        n_nodes = int(cur_node_s.size(0))
        num_nodes.append(n_nodes)

        node_s_list.append(cur_node_s)
        node_v_list.append(cur_node_v)
        edge_s_list.append(cur_edge_s)
        edge_v_list.append(cur_edge_v)
        edge_index_list.append(cur_edge_index + node_offset)
        batch_index.append(batch_idx)
        batch_list.append(
            torch.full((n_nodes,), batch_idx, dtype=torch.long, device=cur_node_s.device)
        )

        if graph.get("x", None) is not None:
            x_list.append(graph["x"])
        if graph.get("seq", None) is not None:
            seq_list.append(graph["seq"])
        if graph.get("mask", None) is not None:
            mask_list.append(graph["mask"])

        node_offset += n_nodes

    valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)
    if len(node_s_list) == 0:
        return {
            "h_V": None,
            "edge_index": None,
            "h_E": None,
            "batch": None,
            "num_nodes": torch.zeros(0, dtype=torch.long),
            "x": None,
            "seq": None,
            "mask": None,
            "pdb_paths": pdb_paths,
            "protein_ids": protein_ids,
            "valid_mask": valid_mask_tensor,
            "batch_index": torch.zeros(0, dtype=torch.long),
        }

    return {
        "h_V": (torch.cat(node_s_list, dim=0), torch.cat(node_v_list, dim=0)),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "h_E": (torch.cat(edge_s_list, dim=0), torch.cat(edge_v_list, dim=0)),
        "batch": torch.cat(batch_list, dim=0),
        "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
        "x": torch.cat(x_list, dim=0) if len(x_list) == len(graph_list) else None,
        "seq": torch.cat(seq_list, dim=0) if len(seq_list) == len(graph_list) else None,
        "mask": torch.cat(mask_list, dim=0) if len(mask_list) == len(graph_list) else None,
        "pdb_paths": pdb_paths,
        "protein_ids": protein_ids,
        "valid_mask": valid_mask_tensor,
        "batch_index": torch.tensor(batch_index, dtype=torch.long),
    }

def collate_fn_conformation_data(data, ts_key=None):
    data = [d for d in data if d is not None and len(d) == 3 and d[1] is not None]
    if len(data) == 0:
        return None

    names, X, _id = list(zip(*data))
    X = unpack_dict(X)
    row_ids = X.get("row_id", None)

    # Pack base tensors.
    ligand_len = torch.tensor([x[0] for x in X["lens"]]).long()
    pocket_len = torch.tensor([x[1] for x in X["lens"]]).long()
    aa_len     = torch.tensor([x[2] for x in X["lens"]]).long()

    ndata = torch.nn.utils.rnn.pad_sequence(X["ndata"], batch_first=True)
    B, max_N, _ = ndata.size()

    edata = None
    if X["edata"][0] is not None:
        edata = torch.cat([pad_edge_type_unsqueeze(x, max_N) for x in X["edata"]])

    # Build ground-truth distances.
    gt_xyz = torch.cat([pad_1D(x, max_N) for x in X["gt_xyz"]]).float()
    D_mask = (gt_xyz.sum(dim=-1) == 0.)[:, None, :].repeat(1, max_N, 1)

    D = torch.cdist(gt_xyz, gt_xyz)
    D = torch.where(D_mask, torch.tensor(-1., device=D.device), D).unsqueeze(-1)

    # Build attention bias.
    attn_bias = get_attn_bias(
        D.squeeze(-1), ndata, edata,
        dist_cut_off=True, aa_reaction_bias=True
    )

    # Build masks.
    pair_mask, pocket_mask = get_pair_and_pocket_mask(D, ligand_len, pocket_len, aa_len)
    ligand_mask = get_ligand_mask(D, ligand_len)

    # Build intra-ligand distances.
    xyz = X["gt_xyz"]
    uff_xyz = torch.cat([pad_1D(x, max_N) for x in xyz]).float()
    uff_xyz = uff_add_pocket_xyz(uff_xyz, gt_xyz, ligand_len)

    intra_D = torch.cdist(uff_xyz, uff_xyz)
    intra_D = torch.where(D_mask, torch.tensor(-1., device=intra_D.device), intra_D).unsqueeze(-1)
    intra_D = intra_D * ~pair_mask

    # Package inputs.
    p_em = stack_if_not_none(X.get("p_em", None))
    pocket_graph = collate_pocket_graphs(X.get("pocket_graph", None))
    pocket_graph_valid = X.get("pocket_graph_valid", None)
    s_em = stack_if_not_none(X.get("s_em", None))
    prod_em = stack_if_not_none(X.get("prod_em", None))
    tsr_em = stack_if_not_none(X.get("tsr_em", None))
    tsr_mask = stack_if_not_none(X.get("tsr_mask", None))
    rxn_fp_em = stack_if_not_none(X.get("rxn_fp_em", None))

    X_new = {
        "x": ndata,
        "edata": edata,
        "D": D,
        "ligand_len": ligand_len,
        "pocket_len": pocket_len,
        "aa_len": aa_len,
        "pocket_mask": pocket_mask,
        "pair_mask": pair_mask,
        "attn_bias": attn_bias,
        "ligand_mask": ligand_mask,
        "atom_dict": X.get("atom_dict", None),
        "aa_dict": X.get("aa_dict", None),
        "target": X.get("target", None),
        "ts_key": ts_key,
        "intra_D": intra_D,
        'tsr_em': tsr_em,
        "tsr_mask": tsr_mask.bool() if tsr_mask is not None else None,
        "p_em": p_em,
        "pocket_graph": pocket_graph,
        "pocket_graph_valid": torch.tensor(pocket_graph_valid, dtype=torch.bool) if pocket_graph_valid is not None else None,
        "s_em": s_em,
        "prod_em": prod_em,
        "rxn_fp_em": rxn_fp_em,
        "row_id": row_ids,
    }

    target = X.get("target", None)

    # Return (X_new, Y_aa, Y_r, target, id).

    # pIC50 task.
    if len(X.get("pic50", [])) != 0:
        # Keep pic50 in X_new for downstream use.
        Y_pic50 = torch.tensor(X["pic50"]).view(-1, 1)
        X_new["pic50"] = Y_pic50
        return X_new, None, None, target, row_ids

    # Reaction label or active sites.
    if len(X.get("label", [])) != 0:
        Y_r = torch.tensor(X["label"]).float().view(-1, 1)

        # Some datasets may not provide asites_*.
        Y_aa = {
            "asites_label": X.get("asites_label", None),
            "active_sites": X.get("active_sites", None),
            "asites_valid": X.get("asites_valid", None),
        }

        # Keep convenience fields.
        X_new["is_substrate"] = X["label"]
        X_new["active_sites"] = X.get("active_sites", None)

        return X_new, Y_aa, Y_r, target, row_ids

    # No label.
    return X_new, None, None, target, row_ids


def collate_fn_no_struct(data, ts_key=None):
    """Return X_new, Y_aa, Y_r, target, and id."""
    names, X, _id = list(zip(*data))
    targets    = [d["target"] for d in X]
    row_ids = [d.get("row_id", idx) for d, idx in zip(X, _id)]
    labels = [d.get("label", None) for d in X]
    X = unpack_dict(X)
    p_em = stack_if_not_none(X.get("p_em", None))
    pocket_graph = collate_pocket_graphs(X.get("pocket_graph", None))
    pocket_graph_valid = X.get("pocket_graph_valid", None)
    s_em = stack_if_not_none(X.get("s_em", None))
    prod_em = stack_if_not_none(X.get("prod_em", None))
    tsr_em = stack_if_not_none(X.get("tsr_em", None))
    tsr_mask = stack_if_not_none(X.get("tsr_mask", None))
    rxn_fp_em = stack_if_not_none(X.get("rxn_fp_em", None))

    X_new = {
        "target": targets,
        "p_em": p_em,
        "pocket_graph": pocket_graph,
        "pocket_graph_valid": torch.tensor(pocket_graph_valid, dtype=torch.bool) if pocket_graph_valid is not None else None,
        "s_em": s_em,
        "prod_em": prod_em,
        "tsr_em": tsr_em,
        "tsr_mask": tsr_mask.bool() if tsr_mask is not None else None,
        "rxn_fp_em": rxn_fp_em,
        "ts_key": ts_key,
        "row_id": row_ids,
    }

    if labels[0] is not None:
        Y_r = torch.tensor(labels).float().view(-1, 1)
    else:
        Y_r = None

    Y_aa = None
    return X_new, Y_aa, Y_r, targets, row_ids
