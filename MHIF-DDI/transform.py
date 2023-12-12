import torch
from torch_geometric.data import Data
from core.fragment_extractors import metis_fragment, random_fragment
import re


def cal_coarsen_adj(subgraphs_nodes_mask):
    mask = subgraphs_nodes_mask.to(torch.float)
    coarsen_adj = torch.matmul(mask, mask.t())
    return coarsen_adj


def to_sparse(node_mask, edge_mask):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]
                      ] = torch.arange(len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected)*num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs


class SubgraphsData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None, **kwargs):
        super().__init__(**kwargs)
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if key == 'edge_index':
            if hasattr(self, 'x_s') and hasattr(self, 'x_t') and self.x_s is not None and self.x_t is not None:
                return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
            else:
                return super().__inc__(key, value, *args, **kwargs)
        elif bool(re.search('(combined_fragments)', key)):
            return getattr(self, key[:-len('combined_fragments')]+'fragments_nodes_mapper').size(0)
        elif bool(re.search('(fragments_batch)', key)):
            return 1+getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_fragments)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)



class FragmentExtrationTransform(object):
    def __init__(self, n_patches, metis=True, drop_rate=0.0, num_hops=1, is_directed=False,patch_num_diff=0):
        super().__init__()
        self.n_patches = n_patches
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_num_diff = patch_num_diff
        self.metis = metis

    def _diffuse(self, A):
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
        RW = A * Dinv
        M = RW
        M_power = M
        # Iterate
        for _ in range(self.patch_num_diff-1):
            M_power = torch.matmul(M_power, M)
        return M_power

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        if self.metis:
            node_masks, edge_masks = metis_fragment(
                data, n_patches=self.n_patches, drop_rate=self.drop_rate, num_hops=self.num_hops, is_directed=self.is_directed)
        else:
            node_masks, edge_masks = random_fragment(
                data, n_patches=self.n_patches, num_hops=self.num_hops)
        fragments_nodes, fragments_edges = to_sparse(node_masks, edge_masks)
        combined_fragments = combine_subgraphs(
            data.edge_index, fragments_nodes, fragments_edges, num_selected=self.n_patches, num_nodes=data.num_nodes)

        if self.patch_num_diff > -1 :
            coarsen_adj = cal_coarsen_adj(node_masks)
            data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        fragments_batch = fragments_nodes[0]
        mask = torch.zeros(self.n_patches).bool()
        mask[fragments_batch] = True
        data.fragments_batch = fragments_batch
        data.fragments_nodes_mapper = fragments_nodes[1]
        data.fragments_edges_mapper = fragments_edges[1]
        data.combined_fragments = combined_fragments
        data.mask = mask.unsqueeze(0)

        data.__num_nodes__ = data.num_nodes  # set number of nodes of the current graph
        return data



