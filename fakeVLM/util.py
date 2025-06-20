import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt


def get_model(model_name, n_classes=2, pretrained=True):
    """Get model architecture based on name"""
    if model_name.startswith('CLIP:'):
        from .clip_models import CLIPModel
        return CLIPModel(model_name, n_classes)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, n_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]