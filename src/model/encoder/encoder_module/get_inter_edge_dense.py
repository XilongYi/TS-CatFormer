def get_energy_inter_edge_dense(intra_edge_dense, batched_data, complex_feat_layer):

    inter_edge_dense = intra_edge_dense
    
    return inter_edge_dense

def get_no_energy_inter_edge_dense(intra_edge_dense, batched_data, complex_feat_layer):

    inter_edge_dense = complex_feat_layer.edge_feat(batched_data['D'], batched_data)

    return inter_edge_dense