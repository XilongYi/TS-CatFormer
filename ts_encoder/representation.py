from __future__ import annotations

import itertools
from enum import IntEnum
from typing import Dict

import torch
from rdkit import Chem
from torch_geometric.data import Data

from .featurizers import atom_features, bond_features


class AtomOriginType(IntEnum):
    REACTANT = 0
    PRODUCT = 1
    DUMMY = 2
    REACTANT_PRODUCT = 3


class EdgeOriginType(IntEnum):
    REACTANT = 0
    PRODUCT = 1
    DUMMY = 2
    REACTANT_PRODUCT = 3


def make_mol(smiles: str):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smiles, params)


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol) -> Dict[int, int]:
    prod_map_to_id = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol_prod.GetAtoms()}
    return {
        atom.GetIdx(): prod_map_to_id[atom.GetAtomMapNum()]
        for atom in mol_reac.GetAtoms()
    }


def generate_bondtype_diff_classes(n: int = 5):
    diff_to_class = {}
    class_id = 0

    def add(diff):
        nonlocal class_id
        diff = tuple(diff)
        if diff not in diff_to_class:
            diff_to_class[diff] = class_id
            class_id += 1

    add([0] * n)
    for j in range(n):
        diff = [0] * n
        diff[j] = 1
        add(diff)
    for i in range(n):
        diff = [0] * n
        diff[i] = -1
        add(diff)
    for i, j in itertools.permutations(range(n), 2):
        diff = [0] * n
        diff[i] = -1
        diff[j] = 1
        add(diff)

    return diff_to_class


def quantize_diff(value: float, eps: float = 1e-6) -> int:
    if abs(value) < eps:
        return 0
    return 1 if value > 0 else -1


class CGRRepresentation:
    def __init__(self, use_bond_change_label: bool = False):
        self.use_bond_change_label = use_bond_change_label
        self.diff_to_class = generate_bondtype_diff_classes() if use_bond_change_label else None

    @staticmethod
    def _feature_and_diff(fn, obj1, obj2) -> list[float]:
        f1 = fn(obj1)
        f2 = fn(obj2)
        return f1 + [b - a for a, b in zip(f1, f2)]

    def __call__(
        self,
        smiles: str,
        extra_atom_features=None,
        **_: dict,
    ) -> Data:
        smiles_reac, smiles_prod = smiles.split(">>", 1)
        mol_reac = make_mol(smiles_reac)
        mol_prod = make_mol(smiles_prod)
        if mol_reac is None or mol_prod is None:
            raise ValueError("invalid reaction smiles")

        ri2pi = map_reac_to_prod(mol_reac, mol_prod)
        n_atoms = mol_reac.GetNumAtoms()

        atom_list = []
        atom_origin_type = []
        bond_change_labels = []
        edge_index = []
        edge_attr = []
        edge_origin_type = []

        for i in range(n_atoms):
            atom_reac = mol_reac.GetAtomWithIdx(i)
            atom_prod = mol_prod.GetAtomWithIdx(ri2pi[i])
            atom_list.append(self._feature_and_diff(atom_features, atom_reac, atom_prod))
            atom_origin_type.append(AtomOriginType.REACTANT_PRODUCT)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                bond_reac = mol_reac.GetBondBetweenAtoms(i, j)
                idx_prod_i = ri2pi[i]
                idx_prod_j = ri2pi[j]
                bond_prod = mol_prod.GetBondBetweenAtoms(idx_prod_i, idx_prod_j)

                if bond_reac is None and bond_prod is None:
                    continue

                bond_feature = self._feature_and_diff(bond_features, bond_reac, bond_prod)
                if self.use_bond_change_label and self.diff_to_class is not None:
                    diff = tuple(quantize_diff(v) for v in bond_feature[11:16])
                    bond_change_labels.append(self.diff_to_class[diff])

                edge_index.extend([(i, j), (j, i)])
                edge_attr.extend([bond_feature, bond_feature])
                edge_origin_type.extend(
                    [EdgeOriginType.REACTANT_PRODUCT, EdgeOriginType.REACTANT_PRODUCT]
                )

        data = Data()
        data.x = torch.tensor(atom_list, dtype=torch.float)
        data.atom_origin_type = torch.tensor(atom_origin_type, dtype=torch.long)
        data.smiles = smiles
        data.num_nodes = n_atoms

        if extra_atom_features is not None:
            data.extra_atom_features = torch.tensor(extra_atom_features, dtype=torch.float)

        if edge_index:
            data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            data.edge_origin_type = torch.tensor(edge_origin_type, dtype=torch.long)
        else:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            data.edge_attr = torch.empty((0, 22), dtype=torch.float)
            data.edge_origin_type = torch.empty((0,), dtype=torch.long)

        if self.use_bond_change_label:
            data.bond_change_label = torch.tensor(bond_change_labels, dtype=torch.long)

        return data

