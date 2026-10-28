import torch.nn as nn

class EnergyHead(nn.Module):
    def __init__(self, atom_type_head, vina_score_head):
        super().__init__()
        self.atom_type_head = atom_type_head
        self.vina_score_head = vina_score_head
        self.supports_seq_only_input = False
        
    
    def forward(self, intra_node, pocket_outputs, intra_edge, batched_data):
        final = []
        x_atom = batched_data['x'].clone()
        _, gscore_loss, gscore_pos_loss, cef_loss = self.vina_score_head(intra_node, intra_edge, x_atom, batched_data)
        final.append(gscore_loss)
        final.append(gscore_pos_loss)
        final.append(cef_loss)
        atom_loss = self.atom_type_head(intra_node, x_atom)
        final.append(atom_loss)
        final.append(None)
        final.append(None)
  

        return final
