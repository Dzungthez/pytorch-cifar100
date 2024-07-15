from training import load_checkpoint, evaluate
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from conf import global_settings as settings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from training import evaluate
from models.attention import attention56
from models.vgg import vgg16_bn
from models.resnet import resnet50 as resnet50_raw
from models.resnet_m import resnet50 as resnet50_m
import dataset


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50_m().to(device)
    checkpoint = torch.load('/teamspace/studios/this_studio/ttt/pytorch-cifar100/scripts/checkpoints/resnet-m/best_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    test_loader = DataLoader(dataset.CangjieDataset(settings.TEST_PATH_CANGJIE), 
        batch_size=128, shuffle=False, num_workers=16, pin_memory=True)
    # Evaluate the model
    test_accuracy, val_top1, val_top5 = evaluate(model, test_loader, device)

    print(f'Accuracy of the network on test dataset: {test_accuracy:.5f}%')
    print(f'Top-1 error: {val_top1:.5f}%')
    print(f'Top-5 error: {val_top5:.5f}%')

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter numbers: {num_params}")