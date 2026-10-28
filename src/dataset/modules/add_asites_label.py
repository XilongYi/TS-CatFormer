from ..utils.data_utils import get_valid_asites_label, atom_dict2_aa_dict

def add_valid_asites_labels(df, new_complex_data):
    proteinid2asite_map = (
        df.groupby("Protein_id")["Active_Sites"]
        .apply(lambda x: list({site for v in x for site in str(v).split('-') if str(v).lower() != 'nan'}))
        .to_dict()
    )
    new_complex_data = get_valid_asites_label(new_complex_data, proteinid2asite_map)
    return new_complex_data

def add_valid_aa_dict_asites_labels(df, new_complex_data):
    for item in new_complex_data:
        if item is not None:
            atom_dict = item['atom_dict']
            aa_dict = atom_dict2_aa_dict(atom_dict)
            item['aa_dict'] = aa_dict
    new_complex_data = add_valid_asites_labels(df, new_complex_data)
    return new_complex_data


def noop_add_asites_labels(df, new_complex_data):
    
    return new_complex_data