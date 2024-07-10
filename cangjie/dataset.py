from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import os
import dotenv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from conf import global_settings as settings
 

class CangjieDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_paths = []
        self.labels = []

        for label in range(len(os.listdir(root_dir))): 
            folder_path = os.path.join(root_dir, str(label))
            for image_name in os.listdir(folder_path):
                if image_name.endswith('.png'):
                    self.image_paths.append(os.path.join(folder_path, image_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)  
        image = image.convert('RGB')
        label = self.labels[idx]

        image = self.transform(image)

        return image, label

if __name__ == '__main__':
    dataset = CangjieDataset(settings.TRAIN_PATH_CANGJIE)
    print(dataset.__len__())