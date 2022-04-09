import torch
import torch.nn as nn
import random
import numpy as np


def set_random_seed(seed: int = 0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including tensor, module...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_n_params(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: model
    :return: int
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float):
    """
    get optimizer
    :param model:
    :param optimizer_name:
    :param learning_rate:
    :param weight_decay:
    :return:
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"wrong value for optimizer {optimizer_name}!")

    return optimizer


def get_lr_scheduler(optimizer: torch.optim, learning_rate: float, t_max: int):
    """
    get learning rate scheduler
    :param optimizer:
    :param learning_rate:
    :param t_max:
    :return:
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=learning_rate / 100)

    return scheduler
