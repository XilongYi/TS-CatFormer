import os.path

import torch
from openbabel import openbabel, pybel
import copy
from collections import defaultdict

from joblib import wrap_non_picklable_objects, delayed
from rdkit import Chem
# from io import BytesIO, StringIO
# from oddt.toolkits.extras.rdkit import fixer
from rdkit import RDLogger
from rdkit.Geometry import Point3D
from .conformation2graph import load_by_rdkit
# from constant import extract_pocket_max_dist, consider_ligand_cofactor
import numpy as np
from scipy.spatial.distance import cdist
from rdkit.Chem import rdmolops
from oddt.toolkits.extras.rdkit import fixer


extract_pocket_max_dist = 7
consider_ligand_cofactor = True
pybel.ob.obErrorLog.SetOutputLevel(0)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def create_ligand_residue(complex):
    for atom in complex.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if res is None:
            symbol = atom.GetSymbol()
            lig_name = 'LIG'
            atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(symbol, residueName=lig_name))
            res = atom.GetPDBResidueInfo()
            res.SetResidueName(lig_name)
            res.SetIsHeteroAtom(True)
        else:
            if res.GetIsHeteroAtom():
                res.SetIsHeteroAtom(False)
    str_ = Chem.MolToPDBBlock(complex)
    pass


def clean_pdb_intersection_code(complex_str):
    complex_str = complex_str.split('\n')
    for i, line in enumerate(complex_str):
        complex_str[i] = line[:26] + ' ' + line[27:]
    return '\n'.join(complex_str)


def rm_water_from_pdb(complex_str):
    final = []
    complex_str = complex_str.split('\n')
    for i, line in enumerate(complex_str):
        resn = line[17:21].strip()
        if resn != 'HOH':
            final.append(line)
    return '\n'.join(final)


def grep_unique_CCD(protein_mol):
    all_CCD = set()
    for atom in protein_mol.GetAtoms():
        resn = atom.GetPDBResidueInfo()
        if resn.GetIsHeteroAtom():
            all_CCD.add(resn.GetResidueName())
    return list(all_CCD)


@wrap_non_picklable_objects
def merge_sdf_pdb_by_residues_near_ligand(complex_pair):
    protein_id, protein_substrate_id, ligand_mols, pocket_mols = complex_pair
    if ligand_mols is None or pocket_mols is None:
        return {'complex_str': None}
    complex = Chem.rdmolops.CombineMols(ligand_mols, pocket_mols)
    create_ligand_residue(complex)
    new_complex = get_residues_near_ligand(complex, n_residues=24)

    # if consider_ligand_cofactor:
    #     append_residues = grep_unique_CCD(pocket_mols)
    # else:
    #     append_residues = [['ZN', 'CL', 'MG']]
    # # Grep only closest xA completed residue
    # try:
    #     pocket, ligand_mols = fixer.ExtractPocketAndLigand(complex, cutoff=extract_pocket_max_dist,
    #                                                   append_residues=append_residues, ligand_residue='LIG',
    #                                                   expandResidues=True)
    # except Exception as e:
    #     # can't find ligand?
    #     print(e, protein_id, "Can't Found Ligand.")
    #     return {'complex_str': None}
    try:
        complex_str = Chem.MolToPDBBlock(new_complex)
    except Exception as e:
        print(f'Cofactor is wrong->{e}, {protein_id}')
        return {'complex_str': None}
    complex_str = clean_pdb_intersection_code(complex_str)
    complex_str = rm_water_from_pdb(complex_str)
    obabel_complex = pybel.readstring('pdb', complex_str)
    complex_str = obabel_complex.write('pdb')
    
    output = {'pdb': protein_id, 'target': protein_substrate_id, 'complex_str': complex_str, 'ligand': ligand_mols}
    return output

@wrap_non_picklable_objects
def merge_sdf_pdb_by_rdkit(complex_pair):
    pdb_code, t, ligand, pdb_mol = complex_pair
    if ligand is None or pdb_mol is None:
        return {'complex_str': None}
    complex = Chem.rdmolops.CombineMols(ligand, pdb_mol)
    create_ligand_residue(complex)
    if consider_ligand_cofactor:
        append_residues = grep_unique_CCD(pdb_mol)
    else:
        append_residues = [['ZN', 'CL', 'MG']]
    try:
        pocket, ligand = fixer.ExtractPocketAndLigand(complex, cutoff=extract_pocket_max_dist,
                                                      append_residues=append_residues, ligand_residue='LIG',
                                                      expandResidues=True)
    except Exception as e:
        print(e, pdb_code, "Can't Found Ligand.")
        return {'complex_str': None}
    try:
        complex = Chem.rdmolops.CombineMols(ligand, pocket)
        complex_str = Chem.MolToPDBBlock(complex)
    except Exception as e:
        print(f'Cofactor is wrong->{e}, {pdb_code}')
        return {'complex_str': None}
    complex_str = clean_pdb_intersection_code(complex_str)
    complex_str = rm_water_from_pdb(complex_str)
    obabel_complex = pybel.readstring('pdb', complex_str)
    complex_str = obabel_complex.write('pdb')
    output = {'pdb': pdb_code, 'target': t, 'complex_str': complex_str, 'ligand': ligand}
    return output





def submol_from_atoms(mol, atom_indices):
    """Extract a submolecule and keep coordinates."""
    atom_indices = sorted(set(atom_indices))
    amap = {old_idx: new_idx for new_idx, old_idx in enumerate(atom_indices)}

    em = Chem.EditableMol(Chem.Mol())
    for idx in atom_indices:
        em.AddAtom(mol.GetAtomWithIdx(idx))

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in amap and a2 in amap:
            em.AddBond(amap[a1], amap[a2], bond.GetBondType())

    new_mol = em.GetMol()

    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        new_conf = Chem.Conformer(len(atom_indices))
        for old_idx, new_idx in amap.items():
            pos = conf.GetAtomPosition(old_idx)
            new_conf.SetAtomPosition(new_idx, pos)
        new_mol.AddConformer(new_conf, assignId=True)

    return new_mol


def get_residues_near_ligand(complex_mol, n_residues=24):
    """Keep the ligand and its nearest residues."""
    conf = complex_mol.GetConformer()

    ligand_atoms = [
        atom.GetIdx()
        for atom in complex_mol.GetAtoms()
        if atom.GetPDBResidueInfo()
        and atom.GetPDBResidueInfo().GetResidueName().strip() == "LIG"
    ]
    if not ligand_atoms:
        raise ValueError("Ligand residue LIG not found")

    ligand_coords = np.array([conf.GetAtomPosition(i) for i in ligand_atoms])

    residues = {}
    for atom in complex_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if not info:
            continue
        res_id = (info.GetChainId(), info.GetResidueNumber(), info.GetResidueName())
        if info.GetResidueName().strip() == "LIG":
            continue
        residues.setdefault(res_id, []).append(atom.GetIdx())

    res_dists = []
    for res_id, atom_indices in residues.items():
        res_coords = np.array([conf.GetAtomPosition(i) for i in atom_indices])
        dmin = cdist(res_coords, ligand_coords).min()
        res_dists.append((res_id, atom_indices, dmin))

    res_dists.sort(key=lambda x: x[2])
    selected_atoms = ligand_atoms[:]
    for res_id, atom_indices, dmin in res_dists[:n_residues]:
        selected_atoms.extend(atom_indices)

    num_atoms = complex_mol.GetNumAtoms()
    selected_atoms = sorted(set([i for i in selected_atoms if i < num_atoms]))
    if not selected_atoms:
        raise ValueError("Selected atom set is empty")

    new_complex = submol_from_atoms(complex_mol, selected_atoms)

    return new_complex

def merge_mols(mol1, mol2):
    new = mol1
    new.OBMol = openbabel.OBMol(mol1.OBMol)
    for a in openbabel.OBMolAtomIter(mol2.OBMol):
        new.OBMol.AddAtom(a)
    for b in openbabel.OBMolBondIter(mol2.OBMol):
        new.OBMol.AddBond(b)
    return new


def load_by_babel(input_file, format='sdf', strip_salt=True):
    if not os.path.exists(input_file):
        return None
    # debug, used to filter timeout obabel pdb
    # pdb_id = sdf_file.split('/')[-1][:4]
    # print(pdb_id, end=',')
    all_mol = defaultdict(list)
    for mol in pybel.readfile(format, input_file):
        # 1. delete all H
        mol.OBMol.DeleteHydrogens()
        # 2. protonaion
        # if format == 'pdb':
        #   mol.OBMol.CorrectForPH()  # this may sometime stuck
        # 3. trip salts
        if strip_salt:
            mol.OBMol.StripSalts()
        # 4. add polar H only
        mol.OBMol.AddPolarHydrogens()
        # 5. add
        all_mol[mol.title].append(mol.write(format))
            # print(mol.title)
        # print(len(mol.atoms), mol.molwt)
        # print("=" * 100)
    if format == 'pdb':
        first_key = list(all_mol.keys())
        if len(first_key) == 0:
            print("@ ERROR PROTEIN FILE", input_file)
            return None
        first_key = first_key[0]
        pdb_id = first_key.split('/')[-1][:4]
        all_mol = all_mol[first_key][0]
        # all_mol = obabel_clean_pdb(all_mol)
    return all_mol


def complex2str(complex_mol):
    c_str = complex_mol.write('pdb').split('\n')
    for i, line in enumerate(c_str):
        if line.find('UNK') != -1:
            c_str[i] = line.replace('ATOM  ', 'HETATM')
    c_str = '\n'.join(c_str)
    return c_str


def write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        x, y, z = new_coords[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()


if __name__ == '__main__':
    import glob
    from gnina_featurizer import obabel_mol_parser, PLIPAtomFeaturizer, PLIPEdgeFeaturizer

    # root = '/opt/home/revoli/data_worker/raw_data/redock'
    # uff_file = 'redock_uff_all'
    root = '/opt/home/revoli/data_worker/v2019-docking'
    uff_file = 'all_uff_ligands'
    # training, 2qbr=910, 7rfs=4WI
    # uff_name = '910'
    # sel_pdb_ids = ['2qbr']
    # uff_name = 'SAM'
    # sel_pdb_ids = ['2uyq']
    uff_name, sel_pdb_ids = '696', ['1o3f']
    # inference
    pdb_ids = []
    for pdb_file in glob.glob(f'{root}/pocket/*.pdb'):
        pdb = os.path.basename(pdb_file)[:4]
        pdb_ids.append(pdb)
    # Uni-Test
    pdb_ids = sel_pdb_ids + pdb_ids[::-1]
    for pdb_id in pdb_ids:
        print(f'->{pdb_id}')
        test_sdf = f'{root}/ligands/{pdb_id}_docked.sdf'
        pdb_file = f'{root}/pocket/{pdb_id}_pocket.pdb'
        ligand_mols = load_by_rdkit(test_sdf)
        pdb_mol = load_by_rdkit(pdb_file, format='pdb')
        # function test
        min_key = [x for x in list(ligand_mols.keys()) if 'min' in x]
        if len(min_key):
            min_key = min_key[0]
        else:
            min_key = list(ligand_mols.keys())[0]
        l_first = ligand_mols[min_key][0]
        # Unit Test
        input = [pdb_id, pdb_id, l_first, pdb_mol]
        merge_data = merge_sdf_pdb_by_rdkit(input)
        merge1_data = merge_sdf_pdb_by_rdkit(input)
        # load uff_ligand

        merge_data['uff_ligand'] = \
            load_by_rdkit(f'/opt/home/revoli/data_worker/v2019-docking/uff/{uff_file}.sdf')[uff_name][0]
        merge1_data['uff_ligand'] = merge_data['uff_ligand']
        # generate feature
        feats = obabel_mol_parser(merge_data, PLIPAtomFeaturizer(), PLIPEdgeFeaturizer(interaction_angle=True),
                                  debug=True)
        feats2 = obabel_mol_parser(merge1_data, PLIPAtomFeaturizer(), PLIPEdgeFeaturizer(interaction_angle=True))
        assert torch.allclose(feats[0], feats2[0])
    # pmap test
    # from data.data_utils import pmap
    # complex_pairs = []
    # for i in range(10):
    #   complex_pairs.append(['1h00', l_first, pdb_mol])
    # res = pmap(merge_sdf_pdb, complex_pairs, n_jobs=10)
    # print(len(res), res[0][0])
    # pass
