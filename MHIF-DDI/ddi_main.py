# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import torch
from ddi_predictor import InteractionPredictor
import torch.multiprocessing as mp
from ddi_train import train as train_tranductive
from ddi_train_inductive import train as train_inductive
import warnings


warnings.filterwarnings("ignore")
dataset_to_abbr = {"drugbank": "drugbank"}
num_node_feats_dict = {"drugbank": 75}
num_ddi_types_dict = {"drugbank": 86}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument(
        "--dataset", type=str, choices=[
            "DrugBank", "ZhangDDI", "ChCh-Miner", "DeepDDI"
        ], default="DrugBank"
    )

    parser.add_argument("--inductive", action="store_true", default=False)
    parser.add_argument("--fold", type=int, choices=[0, 1, 2], default=0)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--pred_mlp_layers", type=int, default=3)

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--patch_num_diff", type=int, default=-1)
    parser.add_argument("--metis_enable", type=bool, default=True)
    parser.add_argument("--n_patches", type=int, default=16)
    parser.add_argument("--metis_drop_rate", type=float, default=0.0)
    parser.add_argument("--metis_num_hops", type=int, default=1)
    parser.add_argument("--model_nlayer_gnn", type=int, default=4)
    parser.add_argument("--model_gnn_type", type=str,choices=["GATConv","GCNConc"], default="GATConv")
    parser.add_argument("--pool", type=str, default="mean")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--min_lr", type=float, default=1e-5)

    args = parser.parse_args()

    args.dataset = dataset_to_abbr[args.dataset.lower()]
    args.num_node_feats = num_node_feats_dict[args.dataset]
    args.num_ddi_types = num_ddi_types_dict[args.dataset]

    return args


if __name__ == "__main__":
    args = main()
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    set_all_seeds(args.seed)
    args.fold = 0
    model = InteractionPredictor(args).to(args.device)

    if not args.inductive:
        train_tranductive(model, args)
    else:
        train_inductive(model, args)



