from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import numpy as np
from torch_geometric.utils import *
import networkx as nx
from tqdm import tqdm
import random
import torch_geometric.transforms as T


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None, lcc=False, save_path=None, args=None, target_node=None):
    val_losses, accs, durations = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: in train_eval:", device)

    data = dataset[0]
    
    pbar = tqdm(range(runs), unit='run')

    for runs_num, _ in enumerate(pbar):
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes, lcc_mask=None, seed=runs_num)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []
        
        if args.lcc:
            path = ("./model/lcc/{}_{}_best_model_run_{}.pth".format(args.dataset, args.model, runs_num))
        else:
            path = ("./model/full/{}_{}_best_model_run_{}.pth".format(args.dataset, args.model, runs_num))
        
        for epoch in range(1, epochs + 1):
            out = train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch

            if logger is not None:
                logger(eval_info)

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

                if args.model_cache:
                    # print("*** Saving Checkpoint ***")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, path)

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        # to print results of this run
        if logger is not None:
            logger.print_statistics(runs_num)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)

    # to print best results of all runs
    if logger is not None:
        logger.print_statistics()

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
    return acc.mean().item()


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    if len(data.y.shape) == 1:
        y = data.y
    else:
        y = data.y.squeeze(1) ## for ogb data

    loss = F.nll_loss(out[data.train_mask], y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    outs = {}

    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        # print("number:", key, len(mask), mask.sum().item())
        # print(key, mask)
        if len(data.y.shape) == 1:
            y = data.y
        else:
            y = data.y.squeeze(1) ## for ogb data

        loss = F.nll_loss(logits[mask], y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc
    return outs
