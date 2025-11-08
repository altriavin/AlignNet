import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from constants import device

from loss_fn import CoMMLoss, RnCLoss, CustomMultiLossLayer
from mmfusion import FusionTransformer
from graph_conv import ComplexGraph


class align_model(nn.Module):
    def __init__(self, args, align_type):
        super().__init__()
        if args.align == "pocket" or align_type == "pocket":
            hidden_dim = args.pocket_hidden_size
            seq_emb_dim, stu_emb_dim = 1280, 3072
            heads, layers = args.pocket_heads, args.pocket_layers

            self.seq_project = nn.Sequential(
                nn.Linear(seq_emb_dim, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(1024, hidden_dim)
            )
            self.stu_project = nn.Sequential(
                nn.Linear(stu_emb_dim, 2560),
                nn.LayerNorm(2560),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(2560, 1280),
                nn.LayerNorm(1280),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(1280, hidden_dim)
            )
            self.eps_seq = nn.Parameter(torch.ones(1, 1, seq_emb_dim) * 0.2)
            self.eps_stu = nn.Parameter(torch.ones(1, 1, stu_emb_dim) * 0.2)
        elif args.align == "ligand" or align_type == "ligand":
            hidden_dim = args.ligand_hidden_size
            seq_emb_dim, stu_emb_dim = 768, 300
            heads, layers = args.ligand_heads, args.ligand_layers
            self.seq_project = nn.Sequential(
                nn.Linear(seq_emb_dim, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(1024, hidden_dim)
            )
            self.stu_project = nn.Sequential(
                nn.Linear(stu_emb_dim, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),
                nn.Linear(512, hidden_dim)
            )
            self.eps_seq = nn.Parameter(torch.ones(1, 1, seq_emb_dim) * 0.2)
            self.eps_stu = nn.Parameter(torch.ones(1, 1, stu_emb_dim) * 0.2)

        self.fusion_transformer = FusionTransformer(hidden_dim, heads, layers, dropout=args.dropout, batch_first=True)
        self.loss_fn = CoMMLoss()

    def data_argument_noisy(self, embeddings, eps):
        random_noise = torch.rand_like(embeddings).to(device)
        embeddings += torch.sign(embeddings) * F.normalize(random_noise, dim=-1) * eps
        return embeddings

    def data_argument_dropout(self, embeddings, masks, dropout_rate=0.15):
        batch_size, seq_len, dim = embeddings.shape
        dropout_mask = torch.ones((batch_size, seq_len), device=device)
        for b in range(batch_size):
            valid_indices = torch.where(masks[b] == 0)[0]
            valid_len = len(valid_indices)
            num_drop = max(1, int(valid_len * dropout_rate))
            if valid_len > 0:
                drop_indices = valid_indices[torch.randperm(valid_len)[:num_drop]]
                dropout_mask[b, drop_indices] = 0.0

        updated_masks = masks.clone()
        updated_masks[dropout_mask == 0] = 1

        embeddings = embeddings * dropout_mask.unsqueeze(-1)
        return embeddings, updated_masks


    def forward_finetune(self, seq_embs, stu_embs):
        seq_embs, seq_masks = seq_embs
        seq_embs = self.seq_project(seq_embs)

        stu_embs, stu_masks = stu_embs
        stu_embs = self.stu_project(stu_embs)

        z = self.fusion_transformer([seq_embs, stu_embs], key_padding_mask=[seq_masks, stu_masks])
        return z

    def forward_finetune_seq(self, seq_embs):
        seq_embs, seq_masks = seq_embs
        seq_embs = self.seq_project(seq_embs)

        z = self.fusion_transformer([seq_embs], key_padding_mask=[seq_masks])
        return z

    def forward_finetune_stu(self, stu_embs):
        stu_embs, stu_masks = stu_embs
        stu_embs = self.stu_project(stu_embs)

        z = self.fusion_transformer([stu_embs], key_padding_mask=[stu_masks])
        return z

    def forward(self, seq_embs, stu_embs):

        seq_embs, seq_masks = seq_embs
        seq_embs_aug_noisy = self.data_argument_noisy(seq_embs, self.eps_seq)
        seq_embs_aug_dropout, seq_masks_aug_dropout = self.data_argument_dropout(seq_embs, seq_masks)

        seq_embs = self.seq_project(seq_embs)
        seq_embs_aug_noisy = self.seq_project(seq_embs_aug_noisy)
        seq_embs_aug_dropout = self.seq_project(seq_embs_aug_dropout)

        stu_embs, stu_masks = stu_embs
        stu_embs_aug_noisy = self.data_argument_noisy(stu_embs, self.eps_stu)
        stu_embs_aug_dropout, stu_masks_aug_dropout = self.data_argument_dropout(stu_embs, stu_masks)

        stu_embs = self.stu_project(stu_embs)
        stu_embs_aug_noisy = self.stu_project(stu_embs_aug_noisy)
        stu_embs_aug_dropout = self.stu_project(stu_embs_aug_dropout)


        z = self.fusion_transformer([seq_embs, stu_embs], key_padding_mask=[seq_masks, stu_masks])

        z1 = self.fusion_transformer([seq_embs], key_padding_mask=[seq_masks])
        z2 = self.fusion_transformer([seq_embs_aug_noisy, stu_embs_aug_noisy], key_padding_mask=[seq_masks, stu_masks])
        z3 = self.fusion_transformer([seq_embs_aug_dropout, stu_embs_aug_dropout], key_padding_mask=[seq_masks_aug_dropout, stu_masks_aug_dropout])
        z4 = self.fusion_transformer([stu_embs], key_padding_mask=[stu_masks])

        # z_x, z_mask = torch.cat([seq_embs, stu_embs], dim=1), torch.cat([seq_masks, stu_masks], dim=1)
        # z = self.fusion_transformer(z_x, key_padding_mask=z_mask)

        # z1 = self.fusion_transformer(seq_embs, key_padding_mask=seq_masks)

        # z2_x, z2_mask = torch.cat([seq_embs_aug_noisy, stu_embs_aug_noisy], dim=1), torch.cat([seq_masks_aug_noisy, stu_masks_aug_noisy], dim=1)
        # z2 = self.fusion_transformer(z2_x, zey_padding_mask=z2_mask)

        # z3_x, z3_mask = torch.cat([seq_embs_aug_dropout, stu_embs_aug_dropout], dim=1), torch.cat([seq_masks_aug_dropout, stu_masks_aug_dropout], dim=1)
        # z3 = self.fusion_transformer(z3_x, key_padding_mask=z3_mask)

        # z4 = self.fusion_transformer(stu_embs, key_padding_mask=stu_masks)

        loss_1, acc_1 = self.loss_fn(z2, z3)
        loss_2_1, acc_2_1 = self.loss_fn(z1, z2)
        loss_2_2, acc_2_2 = self.loss_fn(z1, z3)
        loss_2, acc_2 = (loss_2_1 + loss_2_2) / 2.0, (acc_2_1 + acc_2_2) / 2.0

        loss_3_1, acc_3_1 = self.loss_fn(z4, z2)
        loss_3_2, acc_3_2 = self.loss_fn(z4, z3)
        loss_3, acc_3 = (loss_3_1 + loss_3_2) / 2.0, (acc_3_1 + acc_3_2) / 2.0

        loss_bw = (loss_1 + loss_2 + loss_3) / 3.0
        acc_bw = (acc_1 + acc_2 + acc_3) / 3.0

        return z, loss_bw, acc_bw


class ours_model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_dim = args.ligand_hidden_size
        self.alpha = args.alpha

        self.pocket_align_model = align_model(args, "pocket")
        self.ligand_align_model = align_model(args, "ligand")
        self.fusion_model = FusionTransformer(self.hidden_dim, args.fusion_heads, args.fusion_layers, fusion="x-attn", pool="pool", dropout=args.dropout, batch_first=True)

        self.graph_conv = ComplexGraph(args, self.hidden_dim)

        self.pocket_ligand_align = nn.Sequential(
            nn.Linear(args.pocket_hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(1024, self.hidden_dim)
        )

        self.fc_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 768),
            # nn.LayerNorm(768),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(768, 1)
        )

        self.cl_loss_fn = CoMMLoss()
        self.rnk_loss_fn = RnCLoss(self.hidden_dim * 2)
        # self.mse_loss_fn = nn.MSELoss()
        self.mse_loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)
        self.aggre_loss = CustomMultiLossLayer(3)

    def forward_pred(self, pocket_seq_embs, pocket_stu_embs, ligand_seq_embs, ligand_stu_embs, graphs):
        # pocket_graph, ligand_graph = self.graph_conv(graphs)
        # pocket_z = self.pocket_align_model.forward_finetune(pocket_seq_embs, pocket_stu_embs)
        # ligand_z = self.ligand_align_model.forward_finetune(ligand_seq_embs, ligand_stu_embs)
        # pocket_z = self.pocket_align_model.forward_finetune_seq(pocket_seq_embs)
        # ligand_z = self.ligand_align_model.forward_finetune_seq(ligand_seq_embs)
        pocket_z = self.pocket_align_model.forward_finetune_stu(pocket_stu_embs)
        ligand_z = self.ligand_align_model.forward_finetune_stu(ligand_stu_embs)

        pocket_z = self.pocket_ligand_align(pocket_z)

        # pocket_z = pocket_z + pocket_graph
        # ligand_z = ligand_z + ligand_graph

        fusion_z = self.fusion_model([pocket_z, ligand_z])
        output = self.fc_mlp(fusion_z)
        return output, fusion_z

    def forward(self, pocket_seq_embs, pocket_stu_embs, ligand_seq_embs, ligand_stu_embs, graphs):
        pocket_graph, ligand_graph = self.graph_conv(graphs)
        # print(f'pocket_graph.shape: {pocket_graph}; ligand_graph.shape: {ligand_graph}')

        pocket_z = self.pocket_align_model.forward_finetune(pocket_seq_embs, pocket_stu_embs)
        ligand_z = self.ligand_align_model.forward_finetune(ligand_seq_embs, ligand_stu_embs)

        # pocket_z = self.pocket_align_model.forward_finetune_seq(pocket_seq_embs)
        # ligand_z = self.ligand_align_model.forward_finetune_seq(ligand_seq_embs)
        pocket_z = self.pocket_ligand_align(pocket_z)

        pocket_z = pocket_z + pocket_graph
        ligand_z = ligand_z + ligand_graph

        cl_loss, _ = self.cl_loss_fn(pocket_z, ligand_z)
        # print(f'pocket_z.shpae: {pocket_z.shape}; ligand_z.shape: {ligand_z.shape}; cl_loss: {cl_loss.item()}')
        fusion_z = self.fusion_model([pocket_z, ligand_z])
        output = self.fc_mlp(fusion_z)

        # print(f'fusion_z_loss.shape: {fusion_z_loss.shape}; graphs.label: {graphs.label.unsqueeze(1).shape}')
        rnk_loss = self.rnk_loss_fn(fusion_z, graphs.label.unsqueeze(1))
        # print(f'output.squeeze(): {output}.shape; graphs.label; {graphs.label}')

        mse_loss = self.mse_loss_fn(output, graphs.label.float())

        loss = self.aggre_loss([mse_loss, rnk_loss, cl_loss])

        # loss = mse_loss + self.alpha * rnk_loss + (1 - self.alpha) * cl_loss
        return output, loss, {"mse_loss": mse_loss.item(), "rnk_loss": rnk_loss.item(), "cl_loss": cl_loss.item()}
        # print(f'fusion_z.shape: {fusion_z}')
        # print(f'fusion_z.shape: {fusion_z.shape}')