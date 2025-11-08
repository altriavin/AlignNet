# AlignNet

## AlignNet: Enhancing Protein-Ligand Binding Affinity Prediction through Hierarchical Multi-modal Alignment

This repository provides the official implementation of **AlignNet**, including data processing scripts and a demonstration for training and testing our model.

---

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Complete Workflow](#complete-workflow)
  - [Step 1: Generate Pre-trained Embeddings](#step-1-generate-pre-trained-embeddings)
  - [Step 2: Intra-modal Alignment Training](#step-2-intra-modal-alignment-training)
  - [Step 3: Full Model Training](#step-3-full-model-training)
  - [Step 4: Prediction](#step-4-prediction)
- [Training & Prediction on Your Own Data](#training--prediction-on-your-own-data)
- [Contact](#contact)

---

## Installation

Set up the conda environment using the provided file:

```bash
conda env create -f AlignNet.yaml
conda activate AlignNet
```

---

## Dataset Preparation

Due to file size constraints, this repository includes a small **toy_example** to demonstrate the complete workflow.
The full datasets used in our paper (e.g., **PDBbind**) can be downloaded from their respective official websites. We provide data processing scripts to process the raw data. After our paper is accepted, we will provide all processed datasets used in our work for download.

The toy example is located in the `toy_example/` directory and is pre-configured to work with the provided scripts.

---

## Complete Workflow

This section demonstrates the full pipeline using the provided toy_example dataset.

### Step 1: Generate Pre-trained Embeddings

These scripts are used to process raw protein and ligand files into the necessary input features (embeddings and graphs).

> **Note:**  
> Before running the embedding generation scripts, you need to download the required pre-trained weights for the feature extractors (e.g., ESM, GearNet, Molformer, GraphMVP) from their respective official sources. Please follow the instructions provided by each pre-trained model to obtain and place the weights appropriately.

First, navigate to the processing directory:

```bash
cd get_pretrain_embedding
```

The toy_example for this preprocessing stage is located at `get_pretrain_embedding/toy_example`.
Run the following scripts to generate the required features.

**Generate ESM Embeddings:**

```bash
python -W ignore esm_emb.py
```

**Generate GearNet Embeddings:**

```bash
python -W ignore gearnet_emb.py
```

**Generate Molformer Embeddings:**

```bash
python -W ignore molformer_emb.py
```

**Generate GraphMVP Embeddings:**

```bash
python -W ignore graphmvp_extrator.py
```

**Generate Protein-Ligand Interaction Graph (PyG format):**

```bash
python -W ignore graph_pyg.py
```

After running these scripts, remember to return to the root directory:

```bash
cd ..
```

---

### Step 2: Intra-modal Alignment Training

This stage trains the intra-molecular alignment model of protein and ligand.

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

The resulting aligned models will be saved in the `checkpoint/align_model/` folder.

---

### Step 3: Full Model Training

This command trains the final AlignNet model, loading the pre-trained alignment modules from the previous step.

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

The final model checkpoints will be saved in the `checkpoint/save_model/` folder.

---

### Step 4: Prediction

To predict binding affinities and reproduce our results on the toy dataset, use the following script.

```bash
python -W ignore pred.py \
    --load_model_name repet \
    --dataset toy
```

---

**Note on Model Weights:**  
The alignment weights and the final trained weights for our AlignNet model will be made fully public and available for download upon the publication of our paper. The checkpoints you generate during training will be saved locally in the `checkpoint/` directory.

---

## Training & Prediction on Your Own Data

To use your own dataset, please follow this general guide:

1. **Format Your Data:**
   Structure your dataset of protein-ligand complexes similar to the toy_example. You will typically need **PDB files for proteins** and **MOL2/SDF files for ligands**.

2. **Generate Embeddings:**
   - Place your data in a new folder (e.g., `get_pretrain_embedding/my_dataset`).
   - Modify the paths inside the scripts in the `get_pretrain_embedding/` directory to point to your data.
   - Run the scripts in Step 1 to generate all necessary features.

3. **Train the Model:**
   - Modify the training scripts in Step 2 and Step 3 by changing the `--dataset` argument from `toy` to the name of your dataset.
   - Adjust hyperparameters like `--batch_size`, `--learn_rate`, etc., as needed.

4. **Run Prediction:**
   - Modify the prediction script in Step 4 by changing the `--dataset` argument to your dataset's name and ensuring the `--load_model_name` points to your trained checkpoint.

---