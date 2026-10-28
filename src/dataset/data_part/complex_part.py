from ..utils.data_utils import pmap
from ..utils.gnina_featurizer import obabel_mol_parser
from ..utils.conformation2graph import sdf_load, pdb2mol
from .base_set.lmdb_dataset import LmdbDataset
from .base_set.row_id_dataset import resolve_row_ids_from_df
import gc
import os
import pandas as pd

def require_idx_cols(df: pd.DataFrame):
    required = ["p_idx", "s_idx", "r_idx", 'Substrate_id']
    for c in required:
        if c not in df.columns:
            df[c] = None

def pre_aa_complex(main_set, csv_path, tmp_dir_save, n_jobs=1):

    df = pd.read_csv(csv_path)
    require_idx_cols(df)
    df["row_id"] = resolve_row_ids_from_df(df)

    print(f"[ConformationData] Making complex data {len(df)}")
    if 'Target' not in df.columns:
        df['Target'] = df[main_set.protein_id] + '_' + df[main_set.substrate_id]
    uni_pockets = df['Target'].unique().tolist()
    pocket_mols = read_pocket(main_set, uni_pockets, n_jobs=n_jobs)
    print(f"[ConformationData] Loading Ligands SDF...<-{main_set.ligand_folder}")
    
    df['pose_rank'] = 0
    df = df.astype(dtype={'pose_rank': int})
    ids = df['pose_rank'].tolist()
    sdfs_data = [
        [   
            f"{main_set.work_path}/{main_set.ligand_folder}/{row['Target']}.sdf",
            ids[i],
            f"{row['Target']}"
        ]
        for i, row in df.iterrows()
    ]
    sdf_mols = pmap(sdf_load, sdfs_data, n_jobs=1)

    ligand_mols = {}
    for r in sdf_mols:
        if r:
            ligand_mols[r[1]] = r[0]
    print(f"[ConformationData] Loaded num of substrates: {len(ligand_mols)}/{len(sdf_mols)}\n" + '=' * 100)
    protein_substrate_ids = [f"{row['Target']}" for i, row in df.iterrows()]

    complex_mols = get_merge_pairs(main_set, protein_substrate_ids, ids, ligand_mols, pocket_mols, n_jobs)
    del ligand_mols
    del sdf_mols
    del pocket_mols
    gc.collect()
    
    complex_data = pmap(obabel_mol_parser,
                        complex_mols,
                        node_featurizer=main_set.node_featurizer,
                        edge_featurizer=main_set.edge_featurizer,
                        energy_output_folder=main_set.energy_output_folder,
                        n_jobs=n_jobs)
    
    for item in complex_data:
        if item is not None:
            item['lens'].extend([0, 0])

    valid_ids, data = main_set.get_valid_item(complex_data, protein_substrate_ids)
    print(f"@ [ConformationData] Valid Samples:{len(valid_ids)} / {len(complex_data)}")
    cmoplex_df = df.iloc[valid_ids].reset_index(drop=True)
    data = assign_idx(data, cmoplex_df)
    complex_aa_data = assign_label(main_set, data, cmoplex_df)
    LmdbDataset.write(rows=complex_aa_data, cache_dir=tmp_dir_save, lmdb_name="complex_aa.lmdb")
    ds = LmdbDataset(cache_dir=tmp_dir_save, lmdb_name="complex_aa.lmdb", readonly=True)
    cols = ["label", "Reaction", "r_idx", "s_idx", "p_idx", "Target", "row_id"]
    existing_cols = [c for c in cols if c in cmoplex_df.columns]
    cmoplex_df = cmoplex_df[existing_cols]
    cmoplex_df.to_csv(os.path.join(tmp_dir_save, "complex_df.csv"), index=False)

    return ds, cmoplex_df

def assign_label(main_set, data, df):
    if len(df) > 0:
        data = main_set.assign_label2x(
        main_set.pic50_key, main_set.label_key, main_set.sites_key, df, data)
        return data
    else:
        return None

def assign_idx(data, df):
    assert len(data) == len(df), f"len(data)={len(data)} != len(df)={len(df)}"
    required = ["p_idx", "s_idx", "r_idx", "row_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"df missing columns: {missing}")

    p_arr = df["p_idx"].to_numpy()
    s_arr = df["s_idx"].to_numpy()
    r_arr = df["r_idx"].to_numpy()
    row_id_arr = df["row_id"].to_numpy()
    def to_int_or_none(x):
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return int(x)

    for i in range(len(data)):
        item = data[i]
        if item is None:
            continue
        item["p_idx"] = to_int_or_none(p_arr[i])
        item["s_idx"] = to_int_or_none(s_arr[i])
        item["r_idx"] = to_int_or_none(r_arr[i])
        item["row_id"] = to_int_or_none(row_id_arr[i])

    return data

def check_pocket(pocket_folder, t):
    pocket_f = pocket_folder + f'/{t}.pdb'
    # pocket_f = pocket_folder + f'/{t}.pdb'
    # directly find
    if os.path.exists(pocket_f):
        return pocket_f
    # still not found
    return pocket_f

def read_pocket(main_set, uni_pockets, n_jobs=1):
    # Load pocket PDBs.
    print("Loading Pocket...")
    pocket_folder = f"{main_set.work_path}/{main_set.complex_pocket_folder}"
    count = 0
    mol_pdb_file = [check_pocket(pocket_folder, t) for t in uni_pockets]
    pocket_mols = {}
    tmp_mols = pmap(pdb2mol, mol_pdb_file, n_jobs=n_jobs)
    for i, t in enumerate(uni_pockets):
        if tmp_mols[i]:
            pocket_mols[t] = tmp_mols[i]
            count += 1
    print(f"[ConformationData] Pocket count:{count}/{len(uni_pockets)}\n" + '=' * 100)
    return pocket_mols

def get_merge_pairs(main_set, protein_substrate_ids, ids, ligand_mols, pocket_mols, n_jobs):
    print("[ConformationData] Merging Pocket->Sdf, calculating PLIP.")
    complex_pairs = []
    merge_count = 0
    N = len(ids)
    for i in range(N):
        protein_id = protein_substrate_ids[i]
        protein_substrate_id = protein_substrate_ids[i]
        id = f'{protein_substrate_id}_{ids[i]}'
        try:
            complex_pairs.append([protein_id, protein_substrate_id, ligand_mols[id], pocket_mols[protein_id]])
            merge_count += 1
        except Exception as e:
            complex_pairs.append([None, None, None, None])
            pass
    complex_mols = pmap(main_set.merge_sdf_pdb, complex_pairs, n_jobs=n_jobs)
    
    print(f"[ConformationData] Finished Merging Pocket+Substrate, {merge_count}/{N}\n" + '=' * 100)
    return complex_mols
