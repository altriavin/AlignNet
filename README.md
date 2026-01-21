# AlignNet

## Hierarchical Molecular Alignment for Structure-Agnostic Protein–Ligand Binding Affinity Prediction

This repository provides the official implementation of **AlignNet**, including:

- Data preprocessing scripts (from raw protein/ligand files to embeddings & graphs)
- A **toy_example** for quickly verifying the full pipeline
- Scripts that can preprocess **the full dataset (all files)** (limited by processed data size, please run locally)
- Training & inference code for AlignNet
- Pretrained checkpoints for reproducing results

---

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
  - [1. Download Raw Data](#1-download-raw-data)
  - [2. Generate Pre-trained Embeddings & Graphs](#2-generate-pre-trained-embeddings--graphs)
  - [Resource Notes](#resource-notes)
- [Pretrained Checkpoints (Reproducing Results)](#pretrained-checkpoints-reproducing-results)
- [Training & Inference](#training--inference)
  - [Step 1: Intra-modal Alignment Training](#step-1-intra-modal-alignment-training)
  - [Step 2: Full Model Training](#step-2-full-model-training)
  - [Step 3: Prediction](#step-3-prediction)
- [Training on Your Own Data](#training-on-your-own-data)
- [Contact](#contact)

---

## Installation

Create and activate the conda environment:

```bash
conda env create -f AlignNet.yaml
conda activate AlignNet
```

---

## Dataset Preparation

Due to the **large size of processed features**, this repository does **not** directly host the full processed dataset.  
Instead, we provide:

1. **Raw/original data**
2. A complete preprocessing pipeline that can generate **all required features for the full dataset**
3. A small **toy_example** to demonstrate the entire workflow end-to-end

### 1. Download Raw Data

Raw data can be downloaded here:

```text
https://drive.google.com/drive/folders/1lnJ813-QiMho3duqLKqjTkpXBqkL43NU?usp=sharing
```

### 2. Generate Pre-trained Embeddings & Graphs

This stage converts raw protein/ligand files into required model inputs (embeddings + graphs).

#### Pre-trained weights required

Before running the embedding generation scripts, please download the required pre-trained weights from the official sources and place them according to each project’s instructions:

```text
ESM2:       https://github.com/facebookresearch/esm
GearNet:   https://github.com/DeepGraphLearning/GearNet
Molformer: https://github.com/IBM/molformer
GraphMVP:  https://github.com/chao1224/GraphMVP
```

#### Run preprocessing

Go to the preprocessing directory:

```bash
cd get_pretrain_embedding
```

##### Toy example (quick start)

A small toy dataset is provided at:

- `get_pretrain_embedding/toy_example`

This is intended for quickly testing that the preprocessing + training + prediction pipeline works correctly.

Run the scripts below to generate features:

**Generate ESM Embeddings**
```bash
python -W ignore esm_emb.py
```

**Generate GearNet Embeddings**
```bash
python -W ignore gearnet_emb.py
```

**Generate Molformer Embeddings**
```bash
python -W ignore molformer_emb.py
```

**Generate GraphMVP Embeddings**
```bash
python -W ignore graphmvp_extrator.py
```

**Generate Protein–Ligand Interaction Graph (PyG format)**
```bash
python -W ignore graph_pyg.py
```

Return to the root directory after preprocessing:

```bash
cd ..
```

##### Full dataset processing (all files)

In addition to `toy_example`, the preprocessing scripts are designed to process **the entire dataset (all files)** as long as you point the scripts to the full raw-data directory.

Because the **full processed outputs are very large**, we do not bundle them in this repository. Please run the same preprocessing pipeline on the full dataset locally (and adjust file paths inside the scripts when necessary).

### Resource Notes

- Full preprocessing typically takes **~2 hours**
- It will require **~55 GB** of free disk space for generated intermediate files and outputs  
  (Actual time/disk usage may vary depending on hardware and filesystem performance.)

---

## Model Weights (Reproducing Results)

We provide model weights to help Reproducing the reported results without retraining from scratch. Download link:

```text
https://drive.google.com/drive/folders/1zGTSD1LKtQ8glExy5uD6x_28_ugtl90N?usp=drive_link
```

After downloading, place the checkpoints under the corresponding checkpoint directory (e.g., `checkpoint/align_model/` and/or `checkpoint/save_model/`) and run prediction:

```bash
python -W ignore pred.py \
    --load_model_name <checkpoint_name> \
    --dataset toy
```

> Note: Please ensure the checkpoint filenames / folder structure match the code’s expected loading paths.

---

## Training & Inference

### Step 1: Intra-modal Alignment Training

This stage trains the intra-molecular alignment modules for protein pocket and ligand.

#### Train Protein Alignment Module

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py \
    --align pocket \
    --pocket_layers 4 \
    --pocket_heads 4 \
    --pocket_hidden_size 1024 \
    --save_model_name toy \
    --save_model 1 \
    --log 1
```

#### Train Ligand Alignment Module

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py \
    --align ligand \
    --ligand_layers 4 \
    --ligand_heads 4 \
    --ligand_hidden_size 768 \
    --batch_size 64 \
    --epochs 100 \
    --save_model_name toy \
    --save_model 1
```

Aligned models will be saved to:

- `checkpoint/align_model/`

---

### Step 2: Full Model Training

Train the final AlignNet model (loading alignment modules from Step 1):

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py \
    --dataset toy \
    --load_pocket_model 1 \
    --load_ligand_model 1 \
    --pocket_layers 4 \
    --pocket_heads 4 \
    --pocket_hidden_size 1024 \
    --ligand_layers 4 \
    --ligand_heads 4 \
    --ligand_hidden_size 768 \
    --alpha 0.8 \
    --learn_rate 5e-5 \
    --batch_size 32 \
    --save_model 1 \
    --save_model_name repet \
    --log 1
```

Checkpoints will be saved to:

- `checkpoint/save_model/`

---

### Step 3: Prediction

To predict binding affinities and reproduce results on the toy dataset:

```bash
python -W ignore pred.py \
    --load_model_name repet \
    --dataset toy
```

---

## Training on Your Own Data

1. **Prepare data format**  
   Organize your protein–ligand complexes similarly to `toy_example/` (typically protein PDB + ligand MOL2/SDF).

2. **Generate embeddings & graphs**  
   - Place your raw data under a new folder (e.g., `get_pretrain_embedding/my_dataset`)
   - Update the data paths inside scripts in `get_pretrain_embedding/`
   - Run the preprocessing scripts in the same order as above

3. **Train**  
   - Change `--dataset toy` to your dataset name in training commands
   - Adjust hyperparameters (`--batch_size`, `--learn_rate`, etc.) if needed

4. **Predict**  
   - Set `--dataset` to your dataset name
   - Set `--load_model_name` to your trained checkpoint name

---

## Acknowledgements
This project builds upon state-of-the-art architectures including [ESM2](https://github.com/facebookresearch/esm), [GearNet](https://github.com/DeepGraphLearning/GearNet), [Molformer](https://github.com/IBM/molformer), and [GraphMVP](https://github.com/chao1224/GraphMVP). We gratefully acknowledge the contributions from the respective developers and the research community that made these advancements possible.




## Contact

If you have questions or issues, please open a GitHub Issue in this repository.
