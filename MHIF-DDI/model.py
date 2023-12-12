import torch
import torch.nn as nn
from torch_scatter import scatter
from einops.layers.torch import Rearrange

from core.model_utils.element import MLP
from torch_geometric.nn import GATConv,GCNConv

def get_gnn_model(gnn_type, nhid):
    gnn_model = gnn_type
    hidden_dim = nhid

    if gnn_model == "GATConv":
        return GATConv(hidden_dim, hidden_dim//2, 2)
    elif gnn_model == "GCNConv":
        return GCNConv(hidden_dim,hidden_dim)



class IntraAtt(nn.Module):

    def __init__(self,
                 nhid, nout,
                 nlayer_gnn,
                 gnn_type,
                 dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32):

        super().__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.res = res
        self.gnn = get_gnn_model(gnn_type,nhid)
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])
        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)
        self.output_decoder = MLP(nhid, nout, nlayer=1, with_final_activation=False)

    def forward(self, data,i):
        x = data.x
        if i==0:
            x = x[data.fragments_nodes_mapper]

        edge_index = data.combined_fragments
        batch_x = data.fragments_batch

        fragment = scatter(x, batch_x, dim=0,reduce=self.pooling)[batch_x]
        x = x + self.U[i - 1](fragment)
        x = scatter(x, data.fragments_nodes_mapper,dim=0, reduce=self.pooling)[data.fragments_nodes_mapper]
        x = self.gnn(x,edge_index)


        fragment_x = scatter(x, batch_x, dim=0, reduce=self.pooling)  #pooling

        return fragment_x,x

