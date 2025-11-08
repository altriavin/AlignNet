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

res_type_dict = {
    'ALA': 0, 'GLY': 1, 'SER': 2, 'THR': 3, 'LEU': 4, 'ILE': 5, 'VAL': 6, 'ASN': 7, 'GLN': 8, 'ARG': 9, 'HIS': 10,
    'TRP': 11, 'PHE': 12, 'TYR': 13, 'GLU': 14, 'ASP': 15, 'LYS': 16, 'PRO': 17, 'CYS': 18, 'MET': 19, 'UNK': 20, 
}

protein_letters_1to3 = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "X": "Unk"
}


protein_letters_3to1 = {value.upper(): key for key, value in protein_letters_1to3.items()}

res_type_idx_to_1 = {
    idx: protein_letters_3to1[res_type] for res_type, idx in res_type_dict.items()
}

def quick_pdb_to_seq(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("toto", pdb_path)
    amino_types = []
    for residue in structure.get_residues():
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname()
        if resname.upper() not in res_type_dict:
            resname = 'UNK'
        resname = res_type_dict[resname.upper()]
        amino_types.append(resname)
    amino_types = np.asarray(amino_types, dtype=np.int32)
    return amino_types


def compute_one_esm(pdb_path, outpath=None, model_objs=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_objs is None:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
    else:
        model, batch_converter = model_objs
    model.eval()
    model.to(device)
    name = pdb_path.split('/')[-1][0:-4]
    seq = quick_pdb_to_seq(pdb_path)
    seq = ''.join([res_type_idx_to_1[i] for i in seq])
    # print(f'seq: {seq}')
    tmpdata = [(name, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(tmpdata)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    embed = results['representations'][33][:, 1:-1, :][0]
    embed = embed.cpu()
    print(f'embed.shape: {embed.shape}')
    # if outpath is not None:
    #     torch.save(embed, open(outpath, 'wb'))
    #     # print(f'saving esm embedding to {outpath}')
    # else:
    #     print(f'do not saving the esm embedding!')


def get_esm_emb_dataset():
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_objs = (model, batch_converter)

    # replace this path to your own data paths
    data_sets = ["v2016-all", "v2019-all", "CASF-2016"]
    data_root_path = "../toy_example"
    save_root_path = "../toy_example"
    complex_root_path = "../toy_example"

    for data_set in data_sets:
        affinity_file = os.path.join(data_root_path, f"{data_set}.csv")
        affinity_data = pd.read_csv(affinity_file)
        pdb_ids, affinitys = affinity_data["pdb_id"].values.tolist(), affinity_data["label"].values.tolist()

        for idx, (pdb_id, affinity) in enumerate(tqdm(zip(pdb_ids, affinitys), total=len(pdb_ids))):
            pocket_path = os.path.join(complex_root_path, data_set, pdb_id, f"{pdb_id}_protein.pdb")
            esm_emb_save_path = os.path.join(save_root_path, "esm_protein_emb", f"{pdb_id}_esm.pt")
            compute_one_esm(pocket_path, outpath=esm_emb_save_path, model_objs=model_objs)


if __name__ == "__main__":
    # Example usage
    pocket_path = "toy_example/demo_pocket.pdb"
    output_path = "esm_emb.pt"
    compute_one_esm(pocket_path, outpath=output_path, model_objs=None)

    # get_esm_emb_dataset()