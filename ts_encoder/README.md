# ts_encoder

The core DMPNN architecture is derived from [JohannesKarwou/chemtorch](https://github.com/JohannesKarwou/chemtorch).

## Environment

A simple conda environment that matches the current `ts_encoder` setup:

```bash
conda create -y -n ts_encoder python=3.10
conda activate ts_encoder

pip install rdkit numpy==1.26.4 scikit-learn pandas pyyaml
pip install torch==2.5.1
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv   -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

For GPU usage, install the PyG wheels matching your local PyTorch and CUDA versions:

```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv   -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Notes:
- Replace `TORCH` with your installed PyTorch version, for example `2.5.0` or `2.5.1`.
- Replace `CUDA` with `cpu`, `cu118`, `cu121`, etc.
- Example for CUDA 11.8:

```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv   -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
```

Reference:

```bibtex
@article{landsheere_chemtorch_2025,
  title = {ChemTorch: A Deep Learning Framework for Benchmarking and Developing Chemical Reaction Property Prediction Models},
  doi = {10.26434/chemrxiv-2025-9mggj},
  journal = {ChemRxiv},
  author = {De Landsheere, Jasper and Zamyatin, Anton and Karwounopoulos, Johannes and Heid, Esther},
  year = {2025},
}
```
