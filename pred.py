from tqdm import tqdm
import argparse
import numpy as np
# import torch.optim.lr_scheduler as lr_scheduler
from utils import get_data_loader
import os
import pandas as pd
from constants import data_root_path, pocket_ligand_emb_path, device, align_model_path, save_model_path, pred_checkpoint_path
from sklearn.model_selection import train_test_split
import random
from model import align_model, ours_model
from utils import eval_reg
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from loss_fn import CustomMultiLossLayer

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings
warnings.filterwarnings("ignore")


def eval_model(model, test_loader, args):
    model.eval()
    with torch.no_grad():
        test_targets, test_preds = [], []
        if args.log:
            test_loader = tqdm(test_loader)

        for idx, (pocket_seq_all, pocket_stu_all, ligand_seq_all, ligand_stu_all, graphs) in enumerate(test_loader):

            with autocast():

                pred, _ = model.forward_pred(pocket_seq_all, pocket_stu_all, ligand_seq_all, ligand_stu_all, graphs)
                pred = pred.squeeze(1).cpu().numpy().tolist()
            test_targets.extend(graphs.label.cpu().numpy().tolist())
            test_preds.extend(pred)
        result_reg = eval_reg(test_targets, test_preds)
    return result_reg


def pred_lda(args):

    test_data_path = os.path.join(data_root_path, f"test.csv")
    test_data = pd.read_csv(test_data_path)

    test_data_dataloader = get_data_loader(
        data=test_data, args=args, batch_size=args.batch_size, shuffle=False
    )

    model_path = os.path.join(pred_checkpoint_path, f"{args.dataset}_{args.load_model_name}.pth")
    print(f'model_path: {model_path}')
    model = ours_model(args=args)
    model = model.to(device)
    model_ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(model_ckpt['model_state'], strict=False)
    print(f'loading model from {model_path}')

    result_test = eval_model(model, test_data_dataloader, args)

    print(f'Test RMSE: {result_test["RMSE"]}, PCC: {result_test["PCC"]}, SCC: {result_test["SCC"]}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cv")
    parser.add_argument('--align', type=str, default="cv")

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learn_rate', type=float, default=5e-5)

    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=1.5)
    
    parser.add_argument('--lr_step_size', type=int, default=10, help='step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma for StepLR scheduler')


    parser.add_argument('--max_pocket_seq', type=int, default=1023)
    parser.add_argument('--max_pocket_stu', type=int, default=255)
    parser.add_argument('--max_ligand_seq', type=int, default=199)
    parser.add_argument('--max_ligand_stu', type=int, default=199)

    parser.add_argument('--pocket_layers', type=int, default=4)
    parser.add_argument('--pocket_heads', type=int, default=4)
    parser.add_argument('--ligand_layers', type=int, default=4)
    parser.add_argument('--ligand_heads', type=int, default=4)

    parser.add_argument('--fusion_heads', type=int, default=4)
    parser.add_argument('--fusion_layers', type=int, default=4)

    parser.add_argument('--dropout', type=float, default=0.1)


    parser.add_argument('--pocket_hidden_size', type=int, default=1024)
    parser.add_argument('--ligand_hidden_size', type=int, default=768)
    parser.add_argument('--hidden_size', type=int, default=1024)

    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--log', type=int, default=1)

    parser.add_argument('--load_ligand_model', type=int, default=0)
    parser.add_argument('--load_pocket_model', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--load_model_name', type=str, default="none")

    args = parser.parse_args()

    pred_lda(args)
