# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import LayerNorm
from model import IntraAtt
from einops.layers.torch import Rearrange
from core.model_utils.element import MLP
from core.model_utils.Mutual_Attention import Mutual_Attention
from core.model_utils.Rescal import RESCAL
from core.model_utils.CNNModule import CNNModule


class InteractionPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()

        hidden_dim = args.hidden_dim
        num_node_feats = args.num_node_feats
        num_ddi_types = args.num_ddi_types
        pred_mlp_layers = args.pred_mlp_layers

        dropout = args.dropout

        self.device = args.device

        self.hidden_dim = hidden_dim

        self.num_ddi_types = num_ddi_types

        self.blocks = []
        self.in_feature = args.num_node_feats
        self.initial_norm = LayerNorm(self.in_feature)
        self.KGE = RESCAL(self.num_ddi_types, self.hidden_dim)

        for i in range(args.model_nlayer_gnn):
            block = FR_Block(args, self.in_feature)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.in_feature = self.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + args.n_patches * args.n_patches + num_ddi_types, hidden_dim),
            MLP(hidden_dim, pred_mlp_layers, dropout),
            nn.Linear(hidden_dim, 1)
        )

        if args.inductive:
            self.forward_func = self.forward_inductive
        else:
            self.forward_func = self.forward_transductive

        self.Mutual_Attention = Mutual_Attention(self.hidden_dim)
        self.CNN = CNNModule(args.n_patches)


    def forward_transductive(self, graph_batch_1, graph_batch_2, ddi_type):
        repr_h = []
        repr_t = []

        graph_batch_1.x = self.initial_norm(graph_batch_1.x, graph_batch_1.batch)
        graph_batch_2.x = self.initial_norm(graph_batch_2.x, graph_batch_2.batch)

        for i, block in enumerate(self.blocks):
            out = block(graph_batch_1, graph_batch_2, i)
            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]

            repr_h.append(r_h)
            repr_t.append(r_t)


        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        attention_h = F.elu(F.normalize(self.Mutual_Attention(repr_h, repr_t, flag=True), dim=-1))
        attention_t = F.elu(F.normalize(self.Mutual_Attention(repr_h, repr_t, flag=False), dim=-1))

        attention_h = self.CNN(attention_h)
        attention_t = self.CNN(attention_t)
        attention = attention_h @ attention_t.permute(0, 2, 1)

        attention = F.normalize(attention, dim=-1)
        kge_heads = r_h
        kge_tails = r_t

        ddi_type = torch.tensor(ddi_type).to(self.device)
        # attention = None
        score = self.KGE(kge_heads, kge_tails, ddi_type, attention)

        return score


    def forward_inductive(self, graph_batch_1, graph_batch_2, ddi_type):
        repr_h = []
        repr_t = []

        graph_batch_1.x = self.initial_norm(graph_batch_1.x, graph_batch_1.batch)
        graph_batch_2.x = self.initial_norm(graph_batch_2.x, graph_batch_2.batch)

        for i, block in enumerate(self.blocks):
            out = block(graph_batch_1, graph_batch_2,  i)
            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        attention_h = F.elu(F.normalize(self.Mutual_Attention(repr_h, repr_t, flag=True), dim=-1))
        attention_t = F.elu(F.normalize(self.Mutual_Attention(repr_h, repr_t, flag=False), dim=-1))
        attention_h = self.CNN(attention_h)
        attention_t = self.CNN(attention_t)
        attention = attention_h @ attention_t.permute(0, 2, 1)
        attention = F.normalize(attention, dim=-1)

        ddi_type = torch.tensor(ddi_type).to(self.device)

        kge_heads = r_h
        kge_tails = r_t

        # attention = None
        score = self.KGE(kge_heads, kge_tails, ddi_type, attention)

        return score


class FR_Block(nn.Module):
    def __init__(self, args, input_feature):
        super(FR_Block, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.IntraAtt = IntraAtt(
                                      nhid=self.hidden_dim,
                                      nout=self.hidden_dim,
                                      nlayer_gnn=args.model_nlayer_gnn,
                                      gnn_type=args.model_gnn_type,
                                      pooling=args.pool,
                                      dropout=0,
                                      n_patches=args.n_patches
                                )

        self.reshape = Rearrange('(B p) d ->  B p d', p=args.n_patches)
        self.LN = LayerNorm(self.hidden_dim)
        self.feature_conv = GATConv(input_feature, self.hidden_dim//2, 2)
      

    def forward(self, h_data, t_data, i):

        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)

        h_fragmentRep, h_x = self.IntraAtt(h_data, i)
        t_fragmentRep, t_x = self.IntraAtt(t_data, i)

        h_data.x = F.elu(self.LN(h_x,h_data.fragments_batch))
        t_data.x = F.elu(self.LN(t_x,t_data.fragments_batch))

        h_fragment = self.reshape(h_fragmentRep)
        t_fragment = self.reshape(t_fragmentRep)

        return h_data, t_data, h_fragment, t_fragment
