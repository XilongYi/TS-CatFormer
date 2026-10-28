from __future__ import annotations

from typing import Any, Iterable, Sequence

from rdkit.Chem import Atom, Bond, HybridizationType
from rdkit.Chem.rdchem import BondType


def one_hot_with_unknown(value: Any, options: Sequence[Any]) -> list[int]:
    vec = [0] * (len(options) + 1)
    option_to_index = {option: idx for idx, option in enumerate(options)}
    vec[option_to_index.get(value, len(options))] = 1
    return vec


def _safe_bool_list(values: Iterable[bool]) -> list[int]:
    return [int(v) for v in values]


def atom_features(atom: Atom | None) -> list[float]:
    if atom is None:
        return [0.0] * 44

    features: list[float] = []
    features.extend(
        one_hot_with_unknown(
            atom.GetAtomicNum(),
            [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],
        )
    )
    features.extend(one_hot_with_unknown(atom.GetTotalDegree(), list(range(6))))
    features.extend(one_hot_with_unknown(atom.GetFormalCharge(), [-2, -1, 0, 1, 2]))
    features.extend(one_hot_with_unknown(atom.GetTotalNumHs(), list(range(5))))
    features.extend(
        one_hot_with_unknown(
            atom.GetHybridization(),
            [
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP3,
            ],
        )
    )
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.GetMass()) * 0.01)
    features.extend(
        _safe_bool_list(
            [
                atom.IsInRingSize(3),
                atom.IsInRingSize(4),
                atom.IsInRingSize(5),
                atom.IsInRingSize(6),
                atom.IsInRingSize(7),
            ]
        )
    )
    return features


def bond_features(bond: Bond | None) -> list[float]:
    if bond is None:
        return [0.0] * 11

    features: list[float] = []
    features.extend(
        one_hot_with_unknown(
            bond.GetBondType(),
            [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC],
        )
    )
    features.append(float(bond.GetIsConjugated()))
    features.extend(
        _safe_bool_list(
            [
                bond.IsInRingSize(3),
                bond.IsInRingSize(4),
                bond.IsInRingSize(5),
                bond.IsInRingSize(6),
                bond.IsInRingSize(7),
            ]
        )
    )
    return features

