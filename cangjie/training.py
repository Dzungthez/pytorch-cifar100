import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from conf import global_settings as settings
from utils import get_network, WarmUpLR
import dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

def create_dataloader(data, batch_size, shuffle=True):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, warmup_scheduler, device, epochs=10, writer=None):
    model.train()
    train_loss = []
    train_acc = []
    val_acc = []
    steps = 0
    log_interval = 2000

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Cập nhật warmup scheduler nếu còn trong giai đoạn warmup
                if epoch <= args.warmup:
                    warmup_scheduler.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                steps += 1
                if steps % log_interval == 0:
                    train_accuracy = 100 * correct / total
                    val_accuracy = evaluate(model, val_loader, device)
                    print(f"Step {steps}, Epoch {epoch+1}/{epochs}, Loss: {running_loss / steps:.4f}, "
                          f"Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

                pbar.update(1)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        val_accuracy = evaluate(model, val_loader, device)
        val_acc.append(val_accuracy)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
              f"Val Accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s")

        if writer:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Chỉ cập nhật scheduler nếu đã qua giai đoạn warmup
        if epoch > args.warmup:
            scheduler.step()

    return train_loss, train_acc, val_acc


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    model.train()
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type= bool, default=False, help='use gpu or not')
    parser.add_argument('-batch', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warmup', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')

    args = parser.parse_args()

    model = get_network(args)
    train_data = dataset.CangjieDataset(settings.TRAIN_PATH_CANGJIE)
    val_data = dataset.CangjieDataset(settings.DEV_PATH_CANGJIE)
    test_data = dataset.CangjieDataset(settings.TEST_PATH_CANGJIE)

    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda:0')
        model = model.to(device)
    
    train_loader = create_dataloader(train_data, args.batch)
    val_loader = create_dataloader(val_data, args.batch, shuffle=False)
    test_loader = create_dataloader(test_data, args.batch, shuffle=False)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    warmup_scheduler = WarmUpLR(optimizer, total_iters=len(train_loader) * args.warmup)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)

    writer = SummaryWriter(log_dir=settings.LOG_DIR)

    train_loss, train_acc, val_acc = train(
        model, train_loader, val_loader, loss, optimizer, train_scheduler, warmup_scheduler, device, epochs=settings.EPOCH, writer=writer
    )

    writer.close()

    torch.save(model.state_dict(), os.path.join(settings.CHECKPOINT_PATH, 'final_model.pth'))
