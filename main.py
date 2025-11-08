from tqdm import tqdm
import argparse
import numpy as np
# import torch.optim.lr_scheduler as lr_scheduler
from utils import get_data_loader, get_data_pretrain
import os
import pandas as pd
from constants import data_root_path, pocket_ligand_emb_path, device, align_model_path, save_model_path
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

def train_test_lda(train_loader, val_loader, test_loader, args):
    model = ours_model(args=args)
    model = model.to(device)

    if args.load_pocket_model:
        model_path = f'{align_model_path}/pocket_align_{args.dataset}_{args.pocket_hidden_size}_{args.pocket_heads}_{args.pocket_layers}.pth'
        # print(f'model_pathl: {model_path}')
        model_ckpt = torch.load(model_path, map_location=device)
        model_state_dict = model_ckpt['model_state']
        model.pocket_align_model.load_state_dict(model_state_dict, strict=False)

        # for param in model.pocket_align_model.parameters():
        #     param.requires_grad = False

        print(f'loading pocket model from checkpoint: {model_path} and donot freeze!')
    else:
        print(f'not loading pocket model from checkpoint')

    if args.load_ligand_model:
        model_path = f'{align_model_path}/ligand_align_{args.dataset}_{args.ligand_hidden_size}_{args.ligand_heads}_{args.ligand_layers}.pth'
        model_ckpt = torch.load(model_path, map_location=device)
        model_state_dict = model_ckpt['model_state']
        model.ligand_align_model.load_state_dict(model_state_dict, strict=False)

        # for param in model.ligand_align_model.parameters():
        #     param.requires_grad = False

        print(f'loading ligand model from checkpoint: {model_path} and donot freeze!')
    else:
        print(f'not loading ligand model from checkpoint')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learn_rate
    )
    scaler = GradScaler()

    best_val, best_test = {}, {}
    best_val_rmse = 10000

    for epoch in range(args.epochs):
        model.train()
        if args.log:
            train_loader = tqdm(train_loader, desc=f"training epoch: {epoch}")

        loss_mse, loss_cl, loss_rnk = 0.0, 0.0, 0.0
        for idx, (pocket_seq_embs, pocket_stu_embs, ligand_seq_embs, ligand_stu_embs, graphs) in enumerate(train_loader):

            optimizer.zero_grad()

            with autocast():

                pred, loss, loss_item = model(pocket_seq_embs, pocket_stu_embs, ligand_seq_embs, ligand_stu_embs, graphs)
                loss_mse += loss_item["mse_loss"]
                loss_cl += loss_item["cl_loss"]
                loss_rnk += loss_item["rnk_loss"]

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
        print(f'epoch: {epoch}; mse_loss: {loss_mse}; rnk_loss: {loss_rnk}; cl_loss: {loss_cl}')

        result_val = eval_model(model, val_loader, args)

        if best_val_rmse >= result_val["RMSE"]:
            best_val_rmse = result_val["RMSE"]
            best_val = result_val
            ckpt = {
                'opt': args,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            if args.save_model:
                save_path = f'{save_model_path}/{args.dataset}_{args.save_model_name}.pth'
                torch.save(ckpt, save_path)
                print(f'saving the model to {save_path}')
            best_test = eval_model(model, test_loader, args)

            print(f'*' * 50)
            print(f'epoch: {epoch}')
            print(f'best_test: {best_test}')
            print(f'*' * 50)

    return best_test


def eval_model(model, test_loader, args):
    model.eval()
    with torch.no_grad():
        test_targets, test_preds = [], []
        if args.log:
            test_loader = tqdm(test_loader)

        for idx, (pocket_seq_all, pocket_stu_all, ligand_seq_all, ligand_stu_all, graphs) in enumerate(test_loader):
            with autocast():
                pred, _, _ = model(pocket_seq_all, pocket_stu_all, ligand_seq_all, ligand_stu_all, graphs)
                pred = pred.squeeze(1).cpu().numpy().tolist()
            test_targets.extend(graphs.label.cpu().numpy().tolist())
            test_preds.extend(pred)
        result_reg = eval_reg(test_targets, test_preds)
    return result_reg


def cv_train_test_lda(args):

    train_data_path = os.path.join(data_root_path, f"train.csv")
    train_data = pd.read_csv(train_data_path)

    val_data_path = os.path.join(data_root_path, f"val.csv")
    val_data = pd.read_csv(val_data_path)

    test_data_path = os.path.join(data_root_path, f"test.csv")
    test_data = pd.read_csv(test_data_path)

    train_data_dataloader = get_data_loader(
        data=train_data, args=args, batch_size=args.batch_size, shuffle=True
    )
    val_data_dataloader = get_data_loader(
        data=val_data, args=args, batch_size=args.batch_size, shuffle=False
    )
    test_data_dataloader = get_data_loader(
        data=test_data, args=args, batch_size=args.batch_size, shuffle=False
    )

    best_result = train_test_lda(train_data_dataloader, val_data_dataloader, test_data_dataloader, args)
    print(f'RMSE: {best_result["RMSE"]}; PCC: {best_result["PCC"]}; SCC: {best_result["SCC"]}')


def get_align_model(args):
    train_data_dataloader = get_data_pretrain(
        args=args, batch_size=args.batch_size, shuffle=True
    )
    model = align_model(args=args, align_type=args.align).to(device)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters()}
        ],
        lr=args.learn_rate
    )

    st_epoch = 0

    best_loss = 100000
    for epoch in range(st_epoch, args.epochs + st_epoch):
        model.train()
        if args.log:
            train_data_dataloader = tqdm(train_data_dataloader, desc=f"training epoch: {epoch}")

        loss, acc = [], []
        for idx, (seq_all, stu_all) in enumerate(train_data_dataloader):

            optimizer.zero_grad()

            with autocast():
                z, loss_bw, acc_bw = model(seq_all, stu_all)

                scaler.scale(loss_bw).backward()
                scaler.step(optimizer)
                scaler.update()
            loss.append(loss_bw.item())
            acc.append(acc_bw)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm)

        loss = np.mean(loss)
        acc = np.mean(acc)

        print(f'epoch: {epoch}; loss: {loss}; acc: {acc}')

        if loss <= best_loss:
            best_loss = loss
            ckpt = {
                'opt': args,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            if args.save_model:
                if args.align == "pocket":
                    save_path = f'{align_model_path}/{args.align}_align_{args.save_model_name}_{args.pocket_hidden_size}_{args.pocket_heads}_{args.pocket_layers}.pth'
                else:
                    save_path = f'{align_model_path}/{args.align}_align_{args.save_model_name}_{args.ligand_hidden_size}_{args.ligand_heads}_{args.ligand_layers}.pth'
                torch.save(ckpt, save_path)
                print(f'saving the loss to {save_path}')
            else:
                print(f'donot saving the model!!!')


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
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learn_rate', type=float, default=5e-5)

    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=1.5)
    parser.add_argument('--norm', type=float, default=1.0)

    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.1)

    parser.add_argument('--max_pocket_seq', type=int, default=1023)
    parser.add_argument('--max_pocket_stu', type=int, default=255)
    parser.add_argument('--max_ligand_seq', type=int, default=199)
    parser.add_argument('--max_ligand_stu', type=int, default=199)

    parser.add_argument('--pocket_layers', type=int, default=4)
    parser.add_argument('--pocket_heads', type=int, default=4)
    parser.add_argument('--ligand_layers', type=int, default=4)
    parser.add_argument('--ligand_heads', type=int, default=4)

    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--fusion_heads', type=int, default=4)
    parser.add_argument('--fusion_layers', type=int, default=4)

    parser.add_argument('--pocket_hidden_size', type=int, default=1024)
    parser.add_argument('--ligand_hidden_size', type=int, default=768)
    parser.add_argument('--hidden_size', type=int, default=1024)

    parser.add_argument('--log', type=int, default=1)

    parser.add_argument('--load_ligand_model', type=int, default=0)
    parser.add_argument('--load_pocket_model', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_model_name', type=str, default="none")

    args = parser.parse_args()

    if args.align == "pocket" or args.align == "ligand":
        get_align_model(args)
    else:
        cv_train_test_lda(args)