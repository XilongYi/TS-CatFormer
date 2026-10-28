import os.path
from rdkit import Chem
from collections import defaultdict
from rdkit import RDLogger
from rdkit.Geometry import Point3D
import torch
from Bio.PDB import PDBParser

def load_by_rdkit(input_file, format='sdf', pose=0, warning=True):
    if format == 'sdf':
        try:
            suppl = Chem.SDMolSupplier(input_file, sanitize=True, removeHs=True)
        except OSError as e:
            if warning:
                print(f"# unable to load Sdf file {input_file}")
            return None
        data = defaultdict(list)
        for mol in suppl:
            if mol:
                name = mol.GetProp('_Name')
                if mol.HasProp('minimizedAffinity'):
                    energy = float(mol.GetProp('minimizedAffinity'))
                    if energy > 0:
                        continue
                data[name].append(mol)
        for k, all_mol in data.items():
            multi_poses = []
            if len(all_mol) > 0:
                for i in range(pose + 1):
                    if len(all_mol) > i:
                        my_mol = all_mol[i]
                    else:
                        my_mol = all_mol[0]
                    multi_poses.append(my_mol)
                data[k] = multi_poses
        return data
    



def pdb2mol(input_file, warning=True):
    """Load a PDB file with RDKit."""
    try:
        mol = Chem.MolFromPDBFile(input_file, sanitize=False, removeHs=True)
        if mol is None:
            if warning:
                print(f"# Unable to parse PDB file with RDKit: {input_file}")
            return None
        mol = Chem.RemoveHs(mol, sanitize=False)


        return mol

    except Exception:
        return None
    
def sdf_load(row, warning=True):
    sdf_file, id, protein_substrate_id = row
    id = int(id)
    try:
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=True, removeHs=True)
    except OSError:
        return None
    ligands = list(suppl)
    if id < len(ligands):
        id_name = f'{protein_substrate_id}_{str(id)}'
        return ligands[id], id_name

    return None





