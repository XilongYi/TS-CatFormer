
def get_valid_item_energy(new_complex_data, protein_substrate_ids):
    valid_ids = []
    data = []
    for i, g in enumerate(new_complex_data):
        if g is not None:
            valid_ids.append(i)
            data.append(g)
        # else:
        #     print(protein_substrate_ids[i], end=',')
    return valid_ids, data

def get_valid_item_cef(new_complex_data, protein_substrate_ids):
    valid_ids = []
    data = []
    for i, g in enumerate(new_complex_data):
        if g.get('asites_valid', False) if g is not None else False:
            valid_ids.append(i)
            data.append(g)
        # else:
        #     print(protein_substrate_ids[i], end=',')
    return valid_ids, data

def get_valid_item_asites_crf(new_complex_data, protein_substrate_ids):
    valid_ids = []
    data = []
    for i, g in enumerate(new_complex_data):
        if g is not None:
            valid_ids.append(i)
            data.append(g)

        # else:
        #     print(protein_substrate_ids[i], end=',')
    return valid_ids, data
