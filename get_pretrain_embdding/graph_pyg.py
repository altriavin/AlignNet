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

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l)

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter


def mols2graphs(ligand_path, pocket_path, label, dis_threshold=5.):

    ligand = Chem.MolFromMol2File(ligand_path, removeHs=False)
    pocket = Chem.MolFromPDBFile(pocket_path, removeHs=False)
    
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p = mol2graph(pocket)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
    # edge_index_inter = []
    y = torch.FloatTensor([label])
    pos = torch.concat([pos_l, pos_p], dim=0)
    split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)

    data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, pos=pos, y=y, split=split)

    return data


def get_graph_dataset():
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
            pocket_path = os.path.join(complex_root_path, data_set, pdb_id, f"{pdb_id}_pocket.pdb")
            save_path = os.path.join(save_root_path, "graph", f"{pdb_id}_graph.pyg")
            graph = mols2graphs(ligand_path, pocket_path, 1.0, dis_threshold=5.)
            torch.save(graph, save_path)


if __name__ == "__main__":
    # Example usage
    ligand_path = "toy_example/demo_ligand.mol2"
    pocket_path = "toy_example/demo_pocket.pdb"
    graph = mols2graphs(ligand_path, pocket_path, 1.0, dis_threshold=5.)
    torch.save(graph, "graph.pyg")
    # get_graph_dataset()

