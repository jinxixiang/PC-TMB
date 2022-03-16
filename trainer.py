import torch
import torch.nn as nn
import torch_optimizer as optim
import pandas as pd

# customized libs
import criterions
import models
import datasets


def get_model(conf):
    net = getattr(models, conf.Model.base)
    return net(**conf.Model.params)


def get_loss(conf):
    conf_loss = conf.Loss.base_loss
    assert hasattr(nn, conf_loss.name) or hasattr(criterions, conf_loss.name)
    loss = None
    if hasattr(nn, conf_loss.name):
        loss = getattr(nn, conf_loss.name)
    elif hasattr(criterions, conf_loss.name):
        loss = getattr(criterions, conf_loss.name)

    if len(conf_loss.weight) > 0:
        weight = torch.Tensor(conf_loss.weight)
        conf_loss["weight"] = weight
    return loss(**conf_loss.params)


def get_optimizer(conf):
    conf_optim = conf.Optimizer
    name = conf_optim.optimizer.name
    if hasattr(torch.optim, name):
        optimizer_cls = getattr(torch.optim, name)
    else:
        optimizer_cls = getattr(optim, name)

    if hasattr(conf_optim, "lr_scheduler"):
        scheduler_cls = getattr(torch.optim.lr_scheduler, conf_optim.lr_scheduler.name)
    else:
        scheduler_cls = None
    return optimizer_cls, scheduler_cls


def get_dataset(conf, kfold, mode='train'):
    folds_csv = pd.read_csv(conf.General.folds)

    if conf.General.cross_validation:
        if mode == 'train':
            data_idx = folds_csv[folds_csv['fold'] != kfold].index
        else:
            data_idx = folds_csv[folds_csv['fold'] == kfold].index
    else:
        data_idx = folds_csv[folds_csv['fold'] == mode].index

    name = conf.Data.dataset.name
    dataset_cls = getattr(datasets, name)
    dataset_ = dataset_cls(folds_csv.loc[data_idx].reset_index(drop=True),
                           folds_csv.loc[data_idx].reset_index(drop=True)[conf.General.target_col],
                           conf)

    return dataset_