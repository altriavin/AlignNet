import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
from torch_geometric.data import Batch, Data
import shutil
import gzip
import networkx as nx
from torchdrug.data import Protein
from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import models

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

def get_gearnet_emb(pocket_path, gearnet_emb_save_path):
    gearnet_pretrain_root_path = "angle_gearnet_edge.pth"
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

    gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512],
                                num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                batch_norm=True, concat_hidden=True, short_cut=True, readout="sum").cuda()
    net = torch.load(gearnet_pretrain_root_path)
    gearnet_edge.load_state_dict(net)

    try:
        protein = Protein.from_pdb(pocket_path, atom_feature="position", bond_feature="length", residue_feature="symbol")
        _protein = Protein.pack([protein])
        protein_ = graph_construction_model(_protein)
        protein_.view = 'residue'
        protein_ = protein_.cuda()

        with torch.no_grad():
            gearnet_edge.eval()
            output = gearnet_edge(protein_, protein_.node_feature.float(), all_loss=None, metric=None)

            output['graph_feature'] = output['graph_feature'].cpu()
            output['node_feature'] = output['node_feature'].cpu()
            torch.save(output, gearnet_emb_save_path)
    except Exception as e:
        print(f"Error processing {target}: {e}")
        pass


def get_gearnet_emb_dataset():
    gearnet_pretrain_root_path = "angle_gearnet_edge.pth"
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

    gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512],
                                num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                batch_norm=True, concat_hidden=True, short_cut=True, readout="sum").cuda()
    net = torch.load(gearnet_pretrain_root_path)
    gearnet_edge.load_state_dict(net)

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
            pocket_path = os.path.join(complex_root_path, data_set, pdb_id, f"{pdb_id}_pocket.pdb")
            gearnet_emb_save_path = os.path.join(save_root_path, "gearnet_emb", f"{pdb_id}_gearnet.pt")

            try:
                protein = Protein.from_pdb(pocket_path, atom_feature="position", bond_feature="length", residue_feature="symbol")
                _protein = Protein.pack([protein])
                protein_ = graph_construction_model(_protein)
                protein_.view = 'residue'
                protein_ = protein_.cuda()

                with torch.no_grad():
                    gearnet_edge.eval()
                    output = gearnet_edge(protein_, protein_.node_feature.float(), all_loss=None, metric=None)

                    output['graph_feature'] = output['graph_feature'].cpu()
                    output['node_feature'] = output['node_feature'].cpu()
                    torch.save(output, gearnet_emb_save_path)
            except Exception as e:
                print(f"Error processing {target}: {e}")
                pass


if __name__ == "__main__":
    # Example usage
    demo_pocket_path = "toy_example/demo_pocket.pdb"
    gearnet_emb_save_path = "gearnet_emb.pt"
    get_gearnet_emb(demo_pocket_path, gearnet_emb_save_path)

    # get_gearnet_emb_dataset()
