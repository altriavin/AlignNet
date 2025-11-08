import torch
import esm
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
from torch_geometric.data import Batch, Data
import shutil
import gzip
import networkx as nx
# from torchdrug.data import Protein
# from torchdrug import layers
# from torchdrug.layers import geometry
# from torchdrug import models

from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from scipy.spatial import distance_matrix
import random

from concurrent.futures import ProcessPoolExecutor, as_completed


def get_molformer_emb(smiles, molformer_save_emb_path):
    molformer_tokenizer = AutoTokenizer.from_pretrained("MoLFormer", trust_remote_code=True)
    molformer = AutoModel.from_pretrained("MoLFormer", trust_remote_code=True).cuda()

    molformer_token = molformer_tokenizer(smiles, return_tensors="pt")
    molformer_emb = molformer(molformer_token['input_ids'].cuda(), molformer_token['attention_mask'].cuda()).last_hidden_state
    molformer_emb = molformer_emb.cpu()
    torch.save(molformer_emb, molformer_save_emb_path)


def get_molformer_emb_dataset(smiles, molformer_save_emb_path):
    molformer_tokenizer = AutoTokenizer.from_pretrained("MoLFormer", trust_remote_code=True)
    molformer = AutoModel.from_pretrained("MoLFormer", trust_remote_code=True).cuda()

    data_sets = ["v2016-all", "v2019-all", "CASF-2016"]
    data_root_path = "../toy_example"
    save_root_path = "../toy_example"
    complex_root_path = "../toy_example"

    for data_set in data_sets:
        affinity_file = os.path.join(data_root_path, f"{data_set}.csv")
        affinity_data = pd.read_csv(affinity_file)
        pdb_ids, affinitys = affinity_data["pdb_id"].values.tolist(), affinity_data["label"].values.tolist()

        for idx, (pdb_id, affinity) in enumerate(tqdm(zip(pdb_ids, affinitys), total=len(pdb_ids))):
            ligand_path = os.path.join(complex_root_path, data_set, pdb_id, f"{pdb_id}_ligand.mol2")

            ligand = Chem.MolFromMol2File(ligand_path)
            if ligand is None:
                continue
            smiles = Chem.MolToSmiles(ligand)
            molformer_emb_save_path = os.path.join(save_root_path, "molformer_emb", f"{pdb_id}_molformer.pt")

            molformer_token = molformer_tokenizer(smiles, return_tensors="pt")
            molformer_emb = molformer(molformer_token['input_ids'].cuda(), molformer_token['attention_mask'].cuda()).last_hidden_state
            molformer_emb = molformer_emb.cpu()
            torch.save(molformer_emb, molformer_emb_save_path)


if __name__ == "__main__":
    # Example usage
    smiles = "C1=CC=C(C=C1)C=O"
    molformer_save_emb_path = "molformer_emb.pt"
    get_molformer_emb(smiles, molformer_save_emb_path)

    # get_molformer_emb_dataset()
