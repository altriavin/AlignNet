from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm

from constants import pocket_ligand_emb_path, data_root_path, device
from torch_geometric.data import Batch
import torch

from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve
from scipy.stats import pearsonr, spearmanr

import numpy as np
from Bio.PDB import PDBParser


class PLI_dataset(Dataset):
    def __init__(self, pocket_seq_embs, pocket_stu_embs, ligand_seq_embs, ligand_stu_embs, graphs):
        self.pocket_seq_embs = pocket_seq_embs
        self.pocket_stu_embs = pocket_stu_embs
        self.ligand_seq_embs = ligand_seq_embs
        self.ligand_stu_embs = ligand_stu_embs
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.pocket_seq_embs[idx], self.pocket_stu_embs[idx], self.ligand_seq_embs[idx], self.ligand_stu_embs[idx], self.graphs[idx]


def pad_and_truncate_embeddings(emb_list, batch_size, max_len):
    if batch_size == -1:
        bs = len(emb_list)
    else:
        bs = batch_size
    emb_dim = emb_list[0].size(-1)
    emb_tensor = torch.zeros(bs, max_len, emb_dim, dtype=emb_list[0].dtype, device=emb_list[0].device)
    mask = torch.ones(bs, max_len, dtype=torch.bool, device=emb_list[0].device)

    for i, emb in enumerate(emb_list):
        l = emb.size(0)
        effective_len = min(l, max_len)
        emb_tensor[i, :effective_len] = emb[:effective_len]
        mask[i, :effective_len] = 0
    return emb_tensor, mask


def collate_fn(batch, max_pocket_seq, max_pocket_stu, max_ligand_seq, max_ligand_stu):
    pocket_seq_embs, pocket_stu_embs, ligand_seq_embs, ligand_stu_embs, graphs = zip(*batch)

    pocket_seq_embs, pocket_seq_masks = pad_and_truncate_embeddings(pocket_seq_embs, -1, max_pocket_seq)
    pocket_stu_embs, pocket_stu_masks = pad_and_truncate_embeddings(pocket_stu_embs, -1, max_pocket_stu)
    ligand_seq_embs, ligand_seq_masks = pad_and_truncate_embeddings(ligand_seq_embs, -1, max_ligand_seq)
    ligand_stu_embs, ligand_stu_masks = pad_and_truncate_embeddings(ligand_stu_embs, -1, max_ligand_stu)

   
    graphs = Batch.from_data_list(graphs)
    graphs = graphs.to(device)
    pocket_seq_embs = pocket_seq_embs.to(device)
    pocket_seq_masks = pocket_seq_masks.to(device)
    pocket_stu_embs = pocket_stu_embs.to(device)
    pocket_stu_masks = pocket_stu_masks.to(device)
    ligand_seq_embs = ligand_seq_embs.to(device)
    ligand_seq_masks = ligand_seq_masks.to(device)
    ligand_stu_embs = ligand_stu_embs.to(device)
    ligand_stu_masks = ligand_stu_masks.to(device)

    return (pocket_seq_embs, pocket_seq_masks), (pocket_stu_embs, pocket_stu_masks), (ligand_seq_embs, ligand_seq_masks), (ligand_stu_embs, ligand_stu_masks), graphs


def get_data_loader(data, args, batch_size=256, shuffle=True):
    pdb_ids, labels = data["pdb_id"].values.tolist(), data["label"].values.tolist()

    graphs = []
    if args.log:
        process_data = tqdm(zip(pdb_ids, labels), total=len(pdb_ids))
    else:
        process_data = zip(pdb_ids, labels)
        print(f'dealing data...')

    pocket_seq_embs, pocket_stu_embs, ligand_seq_embs, ligand_stu_embs = [], [], [], []
    for idx, (pdb_id, label) in enumerate(process_data):

        pocket_seq_emb_path = os.path.join(pocket_ligand_emb_path, "esm_protein_emb", f"{pdb_id}_esm.pt")
        pocket_stu_emb_path = os.path.join(pocket_ligand_emb_path, "gearnet_emb", f"{pdb_id}_gearnet.pt")
        ligand_seq_emb_path = os.path.join(pocket_ligand_emb_path, "molformer_emb", f"{pdb_id}_molformer.pt")
        ligand_stu_emb_path = os.path.join(pocket_ligand_emb_path, "graphmvp_emb", f"{pdb_id}_graphmvp.pt")
        graph_path = os.path.join(pocket_ligand_emb_path, "graph_x", f"{pdb_id}_graph_5A_x.pyg")

        if not os.path.exists(pocket_seq_emb_path) or not os.path.exists(pocket_stu_emb_path) or not os.path.exists(ligand_seq_emb_path) or not os.path.exists(ligand_stu_emb_path) or not os.path.exists(graph_path):
            continue

        pocket_seq_emb = torch.load(pocket_seq_emb_path)
        pocket_stu_emb = torch.load(pocket_stu_emb_path)["node_feature"]
        ligand_seq_emb = torch.load(ligand_seq_emb_path)
        ligand_stu_emb = torch.load(ligand_stu_emb_path)["node_feature"]

        pocket_seq_emb = torch.FloatTensor(pocket_seq_emb)
        pocket_stu_emb = torch.FloatTensor(pocket_stu_emb)
        ligand_seq_emb = torch.FloatTensor(ligand_seq_emb).squeeze(0)
        ligand_stu_emb = torch.FloatTensor(ligand_stu_emb).squeeze(0)

        pocket_seq_embs.append(pocket_seq_emb)
        pocket_stu_embs.append(pocket_stu_emb)
        ligand_seq_embs.append(ligand_seq_emb)
        ligand_stu_embs.append(ligand_stu_emb)

        graph = torch.load(graph_path)
        graphs.append(graph)
    PLI_data = PLI_dataset(pocket_seq_embs=pocket_seq_embs, pocket_stu_embs=pocket_stu_embs, ligand_seq_embs=ligand_seq_embs, ligand_stu_embs=ligand_stu_embs, graphs=graphs)
    PLI_dataloader = DataLoader(PLI_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate_fn(batch, args.max_pocket_seq, args.max_pocket_stu, args.max_ligand_seq, args.max_ligand_stu))
    return PLI_dataloader


class feat_align_dataset(Dataset):
    def __init__(self, seq_embs, stu_embs):
        self.seq_embs = seq_embs
        self.stu_embs = stu_embs

    def __len__(self):
        return len(self.stu_embs)

    def __getitem__(self, idx):
        return self.seq_embs[idx], self.stu_embs[idx]


def collate_fn_align(batch, batch_size, max_seq, max_stu):
    seq_embs, stu_embs = zip(*batch)
    seq_embs, seq_embs_mask = pad_and_truncate_embeddings(seq_embs, batch_size, max_seq)
    stu_embs, stu_embs_mask = pad_and_truncate_embeddings(stu_embs, batch_size, max_stu)

    return (seq_embs.to(device), seq_embs_mask.to(device)), (stu_embs.to(device), stu_embs_mask.to(device))


def get_data_pretrain(args, batch_size=256, shuffle=True):
    data = pd.read_csv(os.path.join(data_root_path, "train.csv"))
    pdb_ids, labels = data["pdb_id"].values.tolist(), data["label"].values.tolist()

    if args.log:
        process_data = tqdm(zip(pdb_ids, labels), total=len(pdb_ids))
    else:
        process_data = zip(pdb_ids, labels)
        print(f'dealing data...')

    seq_embs, stu_embs = [], []
    for idx, (pdb_id, label) in enumerate(process_data):
        seq_emb, stu_emb = None, None

        if args.align == "pocket":
            seq_emb_path = os.path.join(pocket_ligand_emb_path, "esm_protein_emb", f"{pdb_id}_esm.pt")
            stu_emb_path = os.path.join(pocket_ligand_emb_path, "gearnet_emb", f"{pdb_id}_gearnet.pt")

            if not os.path.exists(seq_emb_path) or not os.path.exists(stu_emb_path):
                continue

            seq_emb = torch.load(seq_emb_path)
            seq_emb = torch.FloatTensor(seq_emb)
            stu_emb = torch.load(stu_emb_path)["node_feature"]
            stu_emb = torch.FloatTensor(stu_emb)

            max_seq = args.max_pocket_seq
            max_stu = args.max_pocket_stu

        elif args.align == "ligand":

            seq_emb_path = os.path.join(pocket_ligand_emb_path, "molformer_emb", f"{pdb_id}_molformer.pt")
            stu_emb_path = os.path.join(pocket_ligand_emb_path, "graphmvp_emb", f"{pdb_id}_graphmvp.pt")

            if not os.path.exists(seq_emb_path) or not os.path.exists(stu_emb_path):
                continue

            seq_emb = torch.load(seq_emb_path)
            seq_emb = torch.FloatTensor(seq_emb).squeeze(0).detach()
            stu_emb = torch.load(stu_emb_path)["node_feature"]
            stu_emb = torch.FloatTensor(stu_emb).squeeze(0)

            max_seq = args.max_ligand_seq
            max_stu = args.max_ligand_stu

        if seq_emb is not None and stu_emb is not None:
            seq_embs.append(seq_emb)
            stu_embs.append(stu_emb)

    PLI_data = feat_align_dataset(seq_embs=seq_embs, stu_embs=stu_embs)
    PLI_dataloader = DataLoader(PLI_data, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate_fn_align(batch, batch_size, max_seq, max_stu))

    return PLI_dataloader


def eval_reg(test_targets, test_preds):
    mse = mean_squared_error(test_targets, test_preds)
    rmse = np.sqrt(mse)

    pcc, p_value = pearsonr(test_targets, test_preds)
    scc, s_p_value = spearmanr(test_targets, test_preds)

    return {
        'RMSE': rmse,
        'PCC': pcc,
        'SCC': scc
    }
