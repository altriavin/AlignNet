import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch

from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.nn as pyg_nn
import torch_geometric.graphgym.register as register
import math
from constants import device


def _rbf(D, D_min=0., D_max=9., D_count=9):
    
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.to(device)
    # print(f'D_mu: {D_mu}')
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    # print(f'RBF.dtype: {RBF.dtype}')
    
    return RBF


def gnn_norm(x, norm):

    batch_size, num_nodes, num_channels = x.size()
    x = x.view(-1, num_channels)
    x = norm(x)
    x = x.view(batch_size, num_nodes, num_channels)

    return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(drop_rate),
        )
        
    def forward(self, x):
        return self.mlp(x)


class HIL(MessagePassing):
    def __init__(self, input_dim, output_dim, drop_rate, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)
        
        self.mlp_coord = MLP(9, input_dim, 0.0)
        self.out = MLP(input_dim, output_dim, drop_rate)
        
    def message(self, x_j, x_i, radial, index):
        return x_j * radial
    
    def forward(self, x, data, edge_index):
        
        res = x

        pos, size = data.pos, None
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        dist = torch.norm(coord_diff, p=2, dim=-1)
        radial = self.mlp_coord(_rbf(dist))
        # radial = self.mlp_coord(_rbf(dist).to(torch.float16))
        x = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
        x = self.out(x) + res

        return x


class GIGNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(GIGNBlock, self).__init__()

        self.gconv_intra = HIL(input_dim, output_dim, drop_rate)
        self.gconv_inter = HIL(input_dim, output_dim, drop_rate)
        # self.gconv_intra = GCNConvLayer(input_dim, output_dim, drop_rate, True, True)
        # self.gconv_inter = GCNConvLayer(input_dim, output_dim, drop_rate, True, True)

    def forward(self, x, data):
        # print(f'x.dtype: {x.dtype}')
        x_intra = self.gconv_intra(x, data, data.edge_index_intra)
        x_inter = self.gconv_inter(x, data, data.edge_index_inter)
        x = (x_intra + x_inter) / 2

        return x


class DiffPool(nn.Module):
    def __init__(self, input_dim, output_dim, max_num, red_node, edge, drop_rate):
        super().__init__()

        self.max_num = max_num
        self.red_node = red_node
        self.edge = edge
        # self.gnn_p = DenseGCNConv(input_dim, red_node, improved=True, bias=True)
        # self.gnn_p_norm = nn.Sequential(
        #     nn.BatchNorm1d(red_node),
        #     nn.LeakyReLU(0.1),
        # )
        self.gnn_e = DenseGCNConv(input_dim, output_dim, improved=True, bias=True)
        self.gnn_e_norm = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.1),
        )

        # self.out = nn.Linear(output_dim, output_dim)
        # self.out = nn.Linear(output_dim, output_dim)
        # self.out_norm = nn.Sequential(
        #     nn.BatchNorm1d(output_dim),
        # )

    # def pooling(self, x, adj, s, mask=None):

    #     batch_size, num_nodes, _ = x.size()
    #     x = x.unsqueeze(0) if x.dim() == 2 else x
    #     adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    #     s = s.unsqueeze(0) if s.dim() == 2 else s
    #     s = F.softmax(s, dim=-1)

    #     if mask is not None:
    #         mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    #         x, s = x * mask, s * mask

    #     out = torch.matmul(s.transpose(1, 2), x)
    #     out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    #     return out, out_adj, s

    def set_edge_index(self, data, edge):

        switch = {
            "intra": data.edge_index_intra,
            "inter": data.edge_index_inter,
            "intra_lig": data.edge_index_intra_lig,
            "intra_pro": data.edge_index_intra_pro,
        }
        data.edge_index = switch.get(edge, None)

    def forward(self, x, data):

        self.set_edge_index(data, self.edge)
        adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.max_num)
        x, mask = to_dense_batch(x, data.batch, fill_value=0, max_num_nodes=self.max_num)
        # return x, mask
        # adj, mask = adj.to(torch.float16), mask.to(torch.float16)
        # s = gnn_norm(self.gnn_p(x, adj, mask), self.gnn_p_norm)
        # print(f's.shape: {s.shape}')
        # x, adj, s = self.pooling(x, adj, s, mask)
        x = gnn_norm(self.gnn_e(x, adj, mask), self.gnn_e_norm)
        x = torch.mean(x, dim=1)

        return x


class ComplexGraph(nn.Module):
    def __init__(self, args, hidden_dim, drop_rate=0.1):
        super().__init__()

        self.embedding = MLP(35, hidden_dim, 0.0)

        self.GIGNBlock1 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock2 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.GIGNBlock3 = GIGNBlock(hidden_dim, hidden_dim, drop_rate)
        self.diffpool1 = DiffPool(hidden_dim, hidden_dim, 600, args.max_ligand_stu, "intra_lig", drop_rate)
        self.diffpool2 = DiffPool(hidden_dim, hidden_dim, 600, args.max_pocket_stu, "intra_pro", drop_rate)

    def make_edge_index(self, data):
        data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
        data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]

    def forward(self, data):
        x = data.x
        x = self.embedding(x)

        self.make_edge_index(data)
        x = self.GIGNBlock1(x, data)
        x = self.GIGNBlock2(x, data)
        x = self.GIGNBlock3(x, data)

        x_lig = self.diffpool1(x, data)
        x_pro = self.diffpool2(x, data)
        return x_lig, x_pro