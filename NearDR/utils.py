import os
import torch
import random
import numpy as np


def set_seed(seed=2021, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def metric_weights(y_pred, metric_cut):
    y_pred = y_pred.view(-1)
    arr = 1 / torch.arange(1, 1 + len(y_pred)).float().to(y_pred.device)
    if metric_cut is not None:
        arr[metric_cut:] = 0
    weights = torch.abs(arr.view(-1, 1) - arr.view(1, -1))
    return weights


def save_model(model, output_dir, save_name, args, optimizer=None):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))
