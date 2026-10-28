def assign_energy_label2x(pic50_key, label_key, sites_key, df, x):
    assert pic50_key in df.columns, f"DataFrame missing column {pic50_key}"
    for i, row in enumerate(x):
        row['pic50'] = df.iloc[i][pic50_key]
    return x


def assign_cef_label2x(pic50_key, label_key, sites_key, df, x):
    assert label_key in df.columns, f"DataFrame missing column {label_key}"
        
    for i, row in enumerate(x):
        row[label_key] = df.iloc[i][label_key]
        row['active_sites'] = df.iloc[i][sites_key]
    return x




def assign_asites_label2x(pic50_key, label_key, sites_key, df, x):
    assert label_key in df.columns, f"DataFrame missing column {label_key}"
        
    for i, row in enumerate(x):
        row[label_key] = df.iloc[i][label_key]
        row['active_sites'] = df.iloc[i][sites_key]
        for k in ['atom_dict', 'aa_dict']:
            row.pop(k, None)

    return x

def assign_crf_label2x(pic50_key, label_key, sites_key, df, x):
    assert label_key in df.columns, f"DataFrame missing column {label_key}"
    for i, row in enumerate(x):
        row[label_key] = df.iloc[i][label_key]
        for k in ['atom_dict', 'aa_dict']:
            row.pop(k, None)


    return x
        

