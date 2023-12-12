# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from ddi_dataset import DDIDataset, BatchLoader
from torch.utils.data import DataLoader
import csv


def calc_metrics(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    y_pred_label = (y_pred >= 0.5).astype(np.int32)

    acc = metrics.accuracy_score(y_true, y_pred_label)
    auc = metrics.roc_auc_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred_label, zero_division=0)

    p = metrics.precision_score(y_true, y_pred_label, zero_division=0)
    r = metrics.recall_score(y_true, y_pred_label, zero_division=0)
    ap = metrics.average_precision_score(y_true, y_pred)

    return acc, auc, f1, p, r, ap


@torch.no_grad()
def evaluate(model, loader, set_len, criterion, args):
    cur_num = 0
    test_loss = 0.0
    y_pred_all, y_true_all = [], []
    i = 0
    for batch in loader:
        graph_batch_1, graph_batch_2,ddi_type, y_true = batch
        y_true = torch.Tensor(y_true).to(args.device)
        y_pred = model.forward_func(
            graph_batch_1, graph_batch_2, ddi_type
        )
        loss = criterion(y_pred, y_true)
        test_loss += loss.item()

        y_pred_all.append(y_pred.detach().cpu())
        y_true_all.append(y_true.detach().long().cpu())

        cur_num += graph_batch_1.num_graphs // 2
        sys.stdout.write(f"\r{cur_num} / {set_len}")
        sys.stdout.flush()
        i += 1

    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    acc, auc, f1, p, r, ap = calc_metrics(y_pred, y_true)

    return acc, auc, f1, p, r, ap, test_loss / (i + 1)


def train(model, args):
    train_set = DDIDataset(args.dataset, "train", args.fold,args.inductive)
    valid_set = DDIDataset(args.dataset, "valid", args.fold,args.inductive)
    either_set = DDIDataset(args.dataset, "either", args.fold,args.inductive)
    both_set = DDIDataset(args.dataset, "both", args.fold,args.inductive)

    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    either_set_len = len(either_set)
    both_set_len = len(both_set)

    batch_loader = BatchLoader(args)

    train_loader = DataLoader(
        train_set, args.batch_size, True,
        collate_fn=batch_loader.collate_fn, num_workers=3
    )
    valid_loader = DataLoader(
        valid_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn, num_workers=3
    )
    either_loader = DataLoader(
        either_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn, num_workers=3
    )
    both_loader = DataLoader(
        both_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn, num_workers=3
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (1.0 if epoch < 25 else 0.1),
        last_epoch=args.start_epoch - 1
    )

    max_valid_acc, max_either_acc, max_both_acc = 0.0, 0.0, 0.0
    for epoch in range(args.num_epoch):
        print(f"Epoch: {args.start_epoch + epoch}")

        train_loss = 0.0
        cur_num = 0
        y_pred_all, y_true_all = [], []
        train_set.do_shuffle()

        model.train()
        for i, batch in enumerate(train_loader):
            graph_batch_1, graph_batch_2, ddi_type, y_true = batch
            y_true = torch.Tensor(y_true).to(args.device)

            y_pred = model.forward_func(
                graph_batch_1, graph_batch_2, ddi_type
            )
            loss = criterion(y_pred, y_true)
            train_loss += loss.item()

            y_pred_all.append(y_pred.detach().sigmoid().cpu())
            y_true_all.append(y_true.detach().long().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_num += graph_batch_1.num_graphs // 2
            sys.stdout.write(
                f"\r{cur_num} / {train_set_len}, "
                f"{(train_loss / (i + 1)):.6f} "
                "          "
            )
            sys.stdout.flush()
            t_loss = train_loss / (i + 1)
        train_loss = t_loss

        y_pred = torch.cat(y_pred_all)
        y_true = torch.cat(y_true_all)
        train_acc, train_auc, train_f1, train_p, train_r, train_ap = \
            calc_metrics(y_pred, y_true)
        print()
        print(
            f"Train ACC: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}\n"
            f"Train P:   {train_p:.4f}, Train R:   {train_r:.4f}, Train AP: {train_ap:.4f}"
        )

        model.eval()
        valid_acc, valid_auc, valid_f1, valid_p, valid_r, valid_ap, valid_loss = \
            evaluate(model, valid_loader, valid_set_len, criterion, args)
        print()

        print(
            f"Valid ACC:  {valid_acc:.4f}, Valid AUC:  {valid_auc:.4f}, Valid F1:  {valid_f1:.4f}\n"
            f"Valid P:    {valid_p:.4f}, Valid R:    {valid_r:.4f}, Valid AP:  {valid_ap:.4f}"
        )

        either_acc, either_auc, either_f1, either_p, either_r, either_ap, either_loss = \
            evaluate(model, either_loader, either_set_len, criterion, args)
        print()
        print(
            f"Either ACC:  {either_acc:.4f}, Either AUC:  {either_auc:.4f}, Either F1:  {either_f1:.4f}\n"
            f"Either P:    {either_p:.4f}, Either R:    {either_r:.4f}, Either AP:  {either_ap:.4f}"
        )

        both_acc, both_auc, both_f1, both_p, both_r, both_ap, both_loss = \
            evaluate(model, both_loader, both_set_len, criterion, args)
        print()
        print(
            f"Both ACC:    {both_acc:.4f}, Both AUC:    {both_auc:.4f}, Both F1:    {both_f1:.4f}\n"
            f"Both P:      {both_p:.4f}, Both R:      {both_r:.4f}, Both AP:    {both_ap:.4f}"
        )
        print()
        print(
            f"train_loss:   {train_loss:.4f}, valid loss:   {valid_loss:.4f}, either_loss:  {either_loss:.4f}, both_loss:   {both_loss:.4f}"
        )
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            print(f"BEST VALID IN EPOCH {args.start_epoch + epoch}")

        if either_acc > max_either_acc:
            max_either_acc = either_acc
            torch.save(model.state_dict(), f"./model_ind_{args.start_epoch + epoch}.pt")
            print(f"BEST EITHER IN EPOCH {args.start_epoch + epoch}")

        if both_acc > max_both_acc:
            max_both_acc = both_acc
            torch.save(model.state_dict(), f"./model_ind_{args.start_epoch + epoch}.pt")
            print(f"BEST BOTH IN EPOCH {args.start_epoch + epoch}")

        scheduler.step()





