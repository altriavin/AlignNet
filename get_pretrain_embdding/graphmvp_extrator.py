import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '../src_classification')
from os.path import join

# from config import args
# from datasets_complete_feature import MoleculeDatasetComplete
from molecule_gnn_model import GNN_graphpredComplete, GNNComplete
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import (Batch, Data, InMemoryDataset, download_url,
                                  extract_zip)

from rdkit import Chem
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)

# about dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='')
parser.add_argument('--dataset', type=str, default='bace')
parser.add_argument('--num_workers', type=int, default=8)

# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--decay', type=float, default=0)

# about molecule GNN
parser.add_argument('--gnn_type', type=str, default='gin')
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--gnn_lr_scale', type=float, default=1)
parser.add_argument('--model_3d', type=str, default='schnet', choices=['schnet'])

# for AttributeMask
parser.add_argument('--mask_rate', type=float, default=0.15)
parser.add_argument('--mask_edge', type=int, default=0)

# for ContextPred
parser.add_argument('--csize', type=int, default=3)
parser.add_argument('--contextpred_neg_samples', type=int, default=1)

# for SchNet
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--num_interactions', type=int, default=6)
parser.add_argument('--num_gaussians', type=int, default=51)
parser.add_argument('--cutoff', type=float, default=10)
parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add'])
parser.add_argument('--schnet_lr_scale', type=float, default=1)

# for 2D-3D Contrastive CL
parser.add_argument('--CL_neg_samples', type=int, default=1)
parser.add_argument('--CL_similarity_metric', type=str, default='InfoNCE_dot_prod',
                    choices=['InfoNCE_dot_prod', 'EBM_dot_prod'])
parser.add_argument('--T', type=float, default=0.1)
parser.add_argument('--normalize', dest='normalize', action='store_true')
parser.add_argument('--no_normalize', dest='normalize', action='store_false')
parser.add_argument('--SSL_masking_ratio', type=float, default=0)
# This is for generative SSL.
parser.add_argument('--AE_model', type=str, default='AE', choices=['AE', 'VAE'])
parser.set_defaults(AE_model='AE')

# for 2D-3D AutoEncoder
parser.add_argument('--AE_loss', type=str, default='l2', choices=['l1', 'l2', 'cosine'])
parser.add_argument('--detach_target', dest='detach_target', action='store_true')
parser.add_argument('--no_detach_target', dest='detach_target', action='store_false')
parser.set_defaults(detach_target=True)

# for 2D-3D Variational AutoEncoder
parser.add_argument('--beta', type=float, default=1)

# for 2D-3D Contrastive CL and AE/VAE
parser.add_argument('--alpha_1', type=float, default=1)
parser.add_argument('--alpha_2', type=float, default=1)

# for 2D SSL and 3D-2D SSL
parser.add_argument('--SSL_2D_mode', type=str, default='AM')
parser.add_argument('--alpha_3', type=float, default=0.1)
parser.add_argument('--gamma_joao', type=float, default=0.1)
parser.add_argument('--gamma_joaov2', type=float, default=0.1)

# about if we would print out eval metric for training data
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=True)

# about loading and saving
parser.add_argument('--input_model_file', type=str, default='')
parser.add_argument('--output_model_dir', type=str, default='')

# verbosity
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--no_verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=False)

args = parser.parse_args()
print('arguments\t', args)


def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def get_graphmvp_emb(smiles, graphmvp_emb_save_path):
    molecule_model = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    # print(f'molecule_model: {molecule_model}')
    model = GNN_graphpredComplete(args=args, num_tasks=1, molecule_model=molecule_model)
    model.from_pretrained("pretraining_model.pth")
    model.to(device)

    ligand = Chem.MolFromSmiles(smiles)
    try:
        pyg_graph_data = mol_to_graph_data_obj_simple(ligand)
        batch_data = Batch.from_data_list([pyg_graph_data]).to(device)
        output = model(batch_data)
        torch.save(output, open(graphmvp_emb_save_path, 'wb'))
    except:
        print(f'Error processing {pdb_id}_{ligand_idx}, skipping...')
        pass


def get_graphmvp_emb_dataset():
    molecule_model = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    # print(f'molecule_model: {molecule_model}')
    model = GNN_graphpredComplete(args=args, num_tasks=1, molecule_model=molecule_model)
    model.from_pretrained("pretraining_model.pth")
    model.to(device)
    # print(model)

    data_root_path = "../toy_example"
    save_root_path = "../toy_example"
    complex_root_path = "../toy_example"

    data_sets = ["v2016-all", "v2019-all", "CASF-2016"]

    for data_set in data_sets:
        affinity_file = os.path.join(data_root_path, f"{data_set}.csv")
        affinity_data = pd.read_csv(affinity_file)
        filter_affinity_data = affinity_data
        pdb_ids, affinitys = affinity_data["pdb_id"].values.tolist(), affinity_data["label"].values.tolist()

        all_conunt, error_count = 0, 0
        for idx, (pdb_id, affinity) in enumerate(tqdm(zip(pdb_ids, affinitys), total=len(pdb_ids))):
            ligand_path = os.path.join(complex_root_path, data_set, pdb_id, f"{pdb_id}_ligand.mol2")
            output_path = os.path.join(save_root_path, "graphmvp_emb" , f"{pdb_id}_graphmvp.pt")

            ligand = Chem.MolFromMol2File(ligand_path)

            if ligand is None:
                continue
            try:
                pyg_graph_data = mol_to_graph_data_obj_simple(ligand)
                batch_data = Batch.from_data_list([pyg_graph_data]).to(device)
                output = model(batch_data)
                torch.save(output, open(output_path, 'wb'))
            except:
                print(f'Error processing {pdb_id}, skipping...')
                continue


if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Example usage
    smiles = "C1=CC=C(C=C1)C=O"
    graphmvp_emb_save_path = "demo_graphmvp.pt"
    get_graphmvp_emb(smiles, graphmvp_emb_save_path)

    # get_graphmvp_emb_dataset()