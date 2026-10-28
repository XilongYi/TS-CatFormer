# TS-CatFormer

TS-CatFormer is a multimodal framework for enzyme catalytic plasticity prediction. It is designed around the core mechanism of enzyme catalysis: productive enzyme-substrate interaction plus transition-state stabilization. Instead of treating the task as a simple enzyme-substrate binding problem, TS-CatFormer jointly models enzyme information, catalytic microenvironment, and transition-state-aware reaction information to estimate whether an enzyme can catalyze a target reaction.

In the paper, the model combines three main components:

- Global enzyme representation from `ProtT5`, then aligned into a functional space by `ProteinCLIP`.
- Local catalytic microenvironment representation from the enzyme pocket graph encoder.
- Transition-state-aware reaction representation from a pretrained reaction encoder.

These representations are fused through cross-attention and used to output a catalytic plasticity score. The framework supports two practical retrieval settings: ranking candidate enzymes for a query reaction, and ranking candidate reactions for a query enzyme.

## Repository overview
The main paths used for prediction are:

- `scripts/predict_main.py`: main prediction entry.
- `checkpoints/`: CRF checkpoints, reaction encoder checkpoints, and downloaded pretrained model assets.
- `data/`: input data and cached embeddings.
- `src/`: dataset, model, and routine implementations.
- `ts_encoder/`: transition-state-aware reaction encoder code.

## Environment
Use the `tscatformer` environment as the base. For prediction, the packages below are the important ones that are actually used by the current code path.

```bash
conda create -n tscatformer python=3.12
conda activate tscatformer

pip install torch torchvision torchaudio
pip install pytorch-lightning hydra-core omegaconf
pip install pandas numpy tqdm scikit-learn
pip install transformers sentencepiece
pip install onnxruntime
```

Notes:

- `transformers` is used to load `ProtT5` and `MoLFormer`.
- `sentencepiece` is required by the T5 tokenizer.
- `onnxruntime` is required for `ProteinCLIP` inference.
- If you run other structure-aware or training code paths, extra graph and chemistry packages from this repo may also be needed, but the command below mainly depends on the packages listed here.

## Prepare models and data
Recommended order:

1. Download the checkpoint archive and extract it under `checkpoints/`.
   Placeholder link: `https://doi.org/10.5281/zenodo.23684192`

2. Download the Hugging Face model files and place them in the corresponding directories:
   - `checkpoints/MoLFormer-model/`
   - `checkpoints/prot_t5_xl_half_uniref50-enc/`

3. Download the ProteinCLIP model and place it under:
   - `checkpoints/clip/`

4. Download the data archive and extract it under `data/`.
   Placeholder link: `https://doi.org/10.5281/zenodo.23684192`

After preparation, the important layout should look like this:

```text
TSCatformer/
├── checkpoints/
│   ├── MoLFormer-model/
│   ├── prot_t5_xl_half_uniref50-enc/
│   ├── clip/
│   ├── ts_encoder/
│   └── crf_train_enzymecage_global/
├── data/
├── scripts/
├── src/
└── ts_encoder/
```
## System requirements

TS-CatFormer has been tested with the following system configuration:

- Operating system: Ubuntu 22.04 LTS
- Python: 3.12
- GPU: NVIDIA GeForce RTX 3090
- CUDA: 12.6

An NVIDIA GPU with CUDA support is recommended for model inference and training.

## Installation

Create and activate the Conda environment and install the required dependencies as described above.

Typical installation time is approximately 10–20 minutes on a standard workstation, excluding the time required to download pretrained models, checkpoints, and datasets.

## Demo

The example prediction command provided below can be used to run TS-CatFormer on the supplied ZENzyme-18 data.

The expected output is a CSV file containing the model predictions for the input enzyme–reaction pairs.

The expected runtime for the demo is approximately 1 hour on a workstation equipped with an NVIDIA GeForce RTX 3090 GPU.
## Example prediction
Run prediction from the project root:

```bash
python scripts/predict_main.py \
  experiment=predict_crf \
  model.p_in_dim=1024 \
  data.embedding.sequence_output=t5_global_em \
  predict.output_csv_path=ZENzyme-18_out.csv \
  checkpoint.path=checkpoints/crf_train_enzymecage_global/3/epoch-18-test_sr_at_10-0.7045.ckpt
```

This command will load the CRF checkpoint, load `ProtT5`, `ProteinCLIP`, and `MoLFormer` from `checkpoints/`, build the required embeddings, and write the final prediction CSV.
