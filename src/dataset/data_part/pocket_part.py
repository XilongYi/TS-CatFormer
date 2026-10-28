import math
from pathlib import Path
from typing import Iterable

from Bio.PDB import PDBParser
import numpy as np
from omegaconf import ListConfig
import pandas as pd
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric

from ..utils.data_utils import pmap


THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == " ":
            if res.resname not in THREE_TO_ONE:
                if verbose:
                    print(f"Skip non-standard residue: {res}")
                continue
            if (not ensure_ca_exist) or ("CA" in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res["CA"].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(f"Skip hetero residue: {res.full_id}")
    return clean_res_list


def load_clean_residues_from_pdb(pdb_path, ensure_ca_exist=True):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_path)
    return get_clean_res_list(
        structure.get_residues(),
        verbose=False,
        ensure_ca_exist=ensure_ca_exist,
    )


def load_clean_pdb_residue_keys_from_pdb(pdb_path, ensure_ca_exist=True):
    res_list = load_clean_residues_from_pdb(pdb_path, ensure_ca_exist=ensure_ca_exist)
    residue_keys = []
    for res in res_list:
        if res.resname not in THREE_TO_ONE:
            continue
        _, resid, _ = res.full_id[-1]
        residue_keys.append((int(resid), THREE_TO_ONE[res.resname]))
    return residue_keys


def _normalize(tensor, dim=-1):
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


class ProteinGraphDataset:
    def __init__(self, data_list, num_positional_embeddings=16, top_k=30, num_rbf=16, device="cpu"):
        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.letter_to_num = {
            "A": 0,
            "R": 1,
            "N": 2,
            "D": 3,
            "C": 4,
            "Q": 5,
            "E": 6,
            "G": 7,
            "H": 8,
            "I": 9,
            "L": 10,
            "K": 11,
            "M": 12,
            "F": 13,
            "P": 14,
            "S": 15,
            "T": 16,
            "W": 17,
            "Y": 18,
            "V": 19,
        }

    def __getitem__(self, idx):
        return self._featurize_as_graph(self.data_list[idx])

    def _featurize_as_graph(self, protein):
        coords = torch.as_tensor(protein["coords"], device=self.device, dtype=torch.float32)
        seq = torch.as_tensor(
            [self.letter_to_num[a] for a in protein["seq"]],
            device=self.device,
            dtype=torch.long,
        )

        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

        pos_embeddings = self._positional_embeddings(edge_index)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

        dihedrals = self._dihedrals(coords)
        orientations = self._orientations(X_ca)
        sidechains = self._sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num,
            (node_s, node_v, edge_s, edge_v),
        )

        return torch_geometric.data.Data(
            x=X_ca,
            seq=seq,
            name=protein["name"],
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
            mask=mask,
        )

    def _dihedrals(self, X, eps=1e-7):
        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        return torch.cat([torch.cos(D), torch.sin(D)], 1)

    def _positional_embeddings(self, edge_index, num_embeddings=None):
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        return torch.cat((torch.cos(angles), torch.sin(angles)), -1)

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        return -bisector * math.sqrt(1.0 / 3.0) - perp * math.sqrt(2.0 / 3.0)


def residues_to_structure_dict(res_list, name="pocket"):
    res_list = [
        res
        for res in res_list
        if ("N" in res) and ("CA" in res) and ("C" in res) and ("O" in res)
    ]
    structure = {"name": name, "seq": "", "coords": []}
    structure["seq"] = "".join(THREE_TO_ONE[res.resname] for res in res_list)

    for res in res_list:
        res_coords = []
        for atom_name in ["N", "CA", "C", "O"]:
            res_coords.append(list(res[atom_name].coord))
        structure["coords"].append(res_coords)

    return structure


def pdb_to_pocket_graph(pdb_path, device="cpu", top_k=30, num_rbf=16, num_positional_embeddings=16):
    res_list = load_clean_residues_from_pdb(pdb_path, ensure_ca_exist=True)
    structure = residues_to_structure_dict(res_list, name=pdb_path)
    dataset = ProteinGraphDataset(
        [structure],
        num_positional_embeddings=num_positional_embeddings,
        top_k=top_k,
        num_rbf=num_rbf,
        device=device,
    )
    graph = dataset[0]
    return graph, structure


def get_or_build_pocket_embedding(
    main_set,
    sequence_csv_path: str,
    output_name: str = "pocket_gvp_graph",
):
    save_dir = Path(main_set.embedding_save_dir_base) / main_set.dataset_embedding_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pocket_cache_path = save_dir / "pocket_gvp_graph.pt"
    cache = _try_load_pocket_cache(pocket_cache_path)
    if cache is not None:
        return _select_pocket_output(cache, output_name)

    sequence_df = pd.read_csv(sequence_csv_path)
    if "Protein_id" not in sequence_df.columns:
        raise KeyError(f"sequence csv missing required column 'Protein_id': {sequence_csv_path}")
    if "p_idx" in sequence_df.columns:
        expected = list(range(len(sequence_df)))
        got = sequence_df["p_idx"].tolist()
        if got != expected:
            raise ValueError(
                "sequence csv column 'p_idx' must equal row index 0..len(df)-1 for pocket cache"
            )

    pocket_dirs = _normalize_pocket_dirs(main_set.work_path, main_set.protein_only_folder)
    protein_ids = sequence_df["Protein_id"].astype(str).tolist()

    tasks = [
        {
            "p_idx": p_idx,
            "protein_id": protein_id,
            "pocket_dirs": pocket_dirs,
        }
        for p_idx, protein_id in enumerate(protein_ids)
    ]
    results = pmap(
        _build_single_pocket_graph,
        tasks,
        n_jobs=getattr(main_set, "n_jobs", 1),
        verbose=10,
    )

    graphs = []
    pdb_paths = []
    missing_pdb_indices = []
    missing_protein_ids = []
    for result in results:
        graphs.append(result["graph"])
        pdb_paths.append(result["pdb_path"])
        if not result["graph"]["is_valid"]:
            missing_pdb_indices.append(result["graph"]["p_idx"])
            missing_protein_ids.append(result["graph"]["protein_id"])

    if len(graphs) != len(sequence_df):
        raise ValueError(f"len(graphs)={len(graphs)} != len(sequence_df)={len(sequence_df)}")

    cache = {
        "graphs": graphs,
        "protein_ids": protein_ids,
        "pdb_paths": pdb_paths,
        "num_items": len(sequence_df),
        "top_k": 30,
        "output_name": "pocket_gvp_graph",
        "missing_pdb_indices": missing_pdb_indices,
        "missing_protein_ids": missing_protein_ids,
    }
    torch.save(cache, pocket_cache_path)
    return _select_pocket_output(cache, output_name)


def _try_load_pocket_cache(cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        return torch.load(cache_path, map_location="cpu")
    except Exception:
        return None


def _select_pocket_output(cache, output_name: str):
    outputs = {
        "pocket_gvp_graph": cache,
    }
    if output_name not in outputs:
        raise ValueError(
            f"Unknown pocket embedding output '{output_name}'. "
            f"Expected one of: {list(outputs.keys())}"
        )
    return outputs[output_name]


def _normalize_pocket_dirs(work_path: str, pocket_folder) -> list[Path]:
    if pocket_folder is None:
        raise ValueError("main_set.protein_only_folder is required for pocket graph cache")

    raw_dirs: Iterable = (
        list(pocket_folder)
        if isinstance(pocket_folder, (list, tuple, ListConfig))
        else [pocket_folder]
    )
    pocket_dirs = []
    for raw_dir in raw_dirs:
        path = Path(raw_dir)
        if not path.is_absolute():
            path = Path(work_path) / path
        pocket_dirs.append(path)
    return pocket_dirs


def _resolve_pocket_pdb_path(protein_id: str, pocket_dirs: list[Path]) -> Path:
    matches = []
    for pocket_dir in pocket_dirs:
        if not pocket_dir.exists():
            continue
        exact_path = pocket_dir / f"{protein_id}.pdb"
        if exact_path.exists():
            matches.append(exact_path.resolve())
        for path in pocket_dir.glob(f"{protein_id}_*.pdb"):
            matches.append(path.resolve())

    if not matches:
        return None

    matches = sorted(set(matches), key=lambda p: (str(p.parent), p.name))
    return matches[0]


def _build_missing_graph_entry(p_idx: int, protein_id: str):
    return {
        "p_idx": p_idx,
        "protein_id": protein_id,
        "pdb_path": None,
        "is_valid": False,
        "x": None,
        "seq": None,
        "mask": None,
        "node_s": None,
        "node_v": None,
        "edge_index": None,
        "edge_s": None,
        "edge_v": None,
    }


def _build_single_pocket_graph(task: dict):
    p_idx = task["p_idx"]
    protein_id = task["protein_id"]
    pocket_dirs = task["pocket_dirs"]

    pdb_path = _resolve_pocket_pdb_path(protein_id, pocket_dirs)
    if pdb_path is None:
        return {
            "graph": _build_missing_graph_entry(p_idx, protein_id),
            "pdb_path": None,
        }

    graph, _ = pdb_to_pocket_graph(str(pdb_path), device="cpu")
    return {
        "graph": {
            "p_idx": p_idx,
            "protein_id": protein_id,
            "pdb_path": str(pdb_path),
            "is_valid": True,
            "x": graph.x.detach().cpu().contiguous(),
            "seq": graph.seq.detach().cpu().contiguous(),
            "mask": graph.mask.detach().cpu().contiguous(),
            "node_s": graph.node_s.detach().cpu().contiguous(),
            "node_v": graph.node_v.detach().cpu().contiguous(),
            "edge_index": graph.edge_index.detach().cpu().contiguous(),
            "edge_s": graph.edge_s.detach().cpu().contiguous(),
            "edge_v": graph.edge_v.detach().cpu().contiguous(),
        },
        "pdb_path": str(pdb_path),
    }
