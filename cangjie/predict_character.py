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
import models
# from models.resnet import resnet50
from models.attention import attention56
from models.vgg import vgg16_bn
import dataset
import editdistance
from models.resnet import resnet50 as resnet50_raw
from models.resnet_m import resnet50 as resnet50_m
labels_path = '/teamspace/studios/this_studio/ttt/pytorch-cifar100/data/raw/etl_952_singlechar_size_64/952_labels.txt'
def get_cang_len(input_: str):
    if input_ == 'zc': 
        return 1
    return len(input_)

def create_cangjie_mapping(labels_path):
    df = pd.read_csv(labels_path, delim_whitespace=True, header=None, 
                        names=['label', 'character', 'JISx0208', 'UTF8', 'Cangjie'])
    cangjie_mapping = {row['label']: row['Cangjie'] if row['Cangjie'] != 'zc' else '*' for _, row in df.iterrows()}
    return cangjie_mapping

def evaluate_levenstein(mp, model, dataloader, device):
    model.eval()
    numerator_list = []
    denominator_list = []

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc='Evaluating') as pbar:
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                labels = labels.cpu().numpy()
                predicted = predicted.cpu().numpy()

                for i in range(len(labels)):
                    label_mp = mp[str(labels[i])]
                    predicted_mp = mp[str(predicted[i])]
                    cang_len = get_cang_len(label_mp)
                    numerator_list.append(editdistance.eval(predicted_mp, label_mp))
                    denominator_list.append(cang_len)
                pbar.update(1)

    numerator = np.sum(numerator_list)
    denominator = np.sum(denominator_list)
                
    return (1 - numerator / denominator)

if __name__ == '__main__':
    mp = create_cangjie_mapping(labels_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50_raw().to(device)
    checkpoint = torch.load('/teamspace/studios/this_studio/ttt/pytorch-cifar100/scripts/checkpoints/resnet50/checkpoint_epoch_16.pth.tar',
                                    map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    test_loader = DataLoader(dataset.CangjieDataset(settings.TEST_PATH_CANGJIE), 
        batch_size=128, shuffle=False, num_workers=16, pin_memory=True)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter numbers: {num_params}")
    # Evaluate the model
    acc = evaluate_levenstein(mp, model, test_loader, device)
    print(f'Accuracy on the Levenshtein distance metrics: {acc}%')

