import sys
import os
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from conf import global_settings as settings
from utils import get_network, WarmUpLR
import dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def create_dataloader(data, batch_size, shuffle=True, num_workers=16):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

def load_checkpoint(checkpoint_dir, model, optimizer=None, scheduler=None):
    checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpot.pth.tar')
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return 0

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, warmup_scheduler, 
          device, epochs=10, writer=None, start_epoch=0, checkpoint_dir='.', save_interval=5):
    model.train()
    train_loss = []
    train_acc = []
    val_acc = []
    steps = 0
    log_interval = 3000
    best_val_acc = 0.0
    epochs += start_epoch
    for epoch in range(start_epoch, epochs):
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

                if epoch <= args.warmup:
                    warmup_scheduler.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                steps += 1
                if steps % log_interval == 0:
                    train_accuracy = 100 * correct / total
                    val_accuracy, val_top1_err, val_top5_err = evaluate(model, val_loader, device)
                    print(f"Step {steps}, Epoch {epoch+1}/{epochs}, Loss: {running_loss / steps:.4f}, "
                          f"Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

                pbar.update(1)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        val_accuracy, val_top1_err, val_top5_err = evaluate(model, val_loader, device)
        val_acc.append(val_accuracy)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
              f"Val Accuracy: {val_accuracy:.2f}%, Val Top 1 Error: {val_top1_err:.4f}, "
              f"Val Top 5 Error: {val_top5_err:.4f}, Time: {epoch_time:.2f}s")

        if writer:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Top 1 Error/val', val_top1_err, epoch)
            writer.add_scalar('Top 5 Error/val', val_top5_err, epoch)

        # Chỉ cập nhật scheduler nếu đã qua giai đoạn warmup
        if epoch > args.warmup:
            scheduler.step()

        # Lưu checkpoint sau mỗi save_interval epoch
        if epoch % save_interval == 0 or epoch == epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'warmup_scheduler': warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
            }, checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth.tar')

        if val_accuracy > best_val_acc*1.0005:
            best_val_acc = val_accuracy
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'warmup_scheduler': warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
            }, checkpoint_dir, 'best_checkpoint.pth.tar')

    return train_loss, train_acc, val_acc

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, top1_predicted = torch.max(outputs, 1)
            top5_predicted = torch.topk(outputs, 5, dim=1).indices

            total += labels.size(0)
            correct += (top1_predicted == labels).sum().item()
            top1_correct += (top1_predicted == labels).sum().item()
            top5_correct += sum([1 if labels[i] in top5_predicted[i] else 0 for i in range(labels.size(0))])

    top1_err = 1 - top1_correct / total
    top5_err = 1 - top5_correct / total
    accuracy = 100 * correct / total
    model.train()
    return accuracy, top1_err, top5_err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('-batch', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warmup', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints/attention', help='directory to save checkpoints')
    parser.add_argument('-save_interval', type=int, default=5, help='save checkpoint every n epochs')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    model = get_network(args)
    train_data = dataset.CangjieDataset(settings.TRAIN_PATH_CANGJIE)
    val_data = dataset.CangjieDataset(settings.DEV_PATH_CANGJIE)
    test_data = dataset.CangjieDataset(settings.TEST_PATH_CANGJIE)

    device = torch.device('cpu')
    if args.gpu:
        device = torch.device('cuda:0')
        model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter numbers: {num_params}")
    
    train_loader = create_dataloader(train_data, args.batch, num_workers=16)
    val_loader = create_dataloader(val_data, args.batch, shuffle=False, num_workers=16)
    test_loader = create_dataloader(test_data, args.batch, shuffle=False, num_workers=16)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    warmup_scheduler = WarmUpLR(optimizer, total_iters=len(train_loader) * args.warmup)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)

    writer = SummaryWriter(log_dir=settings.LOG_DIR)

    start_epoch = load_checkpoint(args.checkpoint_dir, model, optimizer, train_scheduler)

    train_loss, train_acc, val_acc = train(
        model, train_loader, val_loader, loss, optimizer, train_scheduler, warmup_scheduler, device, 
        epochs=settings.EPOCH, writer=writer, start_epoch=start_epoch, checkpoint_dir=args.checkpoint_dir, 
        save_interval=args.save_interval
    )

    writer.close()

    test_acc, top1_err, top5_err = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Top 1 Error Rate: {top1_err:.4f}")
    print(f"Top 5 Error Rate: {top5_err:.4f}")

    
    
   
