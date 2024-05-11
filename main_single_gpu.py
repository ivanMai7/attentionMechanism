import torch
import torch.nn as nn
from models.ResNet18 import ResNet18
from models.ViT import ViT
from utils import AverageMeter
from dataset import train_dataset
from dataset import val_dataset
from torch.utils.data import DataLoader

def get_device():
    device = torch.device('cuda', 0)
    return device

def accuracy(pred, label):
    pred = pred.argmax(dim=1)
    n = len(pred)
    cnt = 0
    for i in range(n):
        if pred[i] == label[i]:
            cnt += 1
    return cnt / n

def validate(model, dataloader, criterion):
    print(f"--- Validation")
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.eval()
    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        device = get_device()
        image = image.to(device)
        label = label.to(device)

        out = model(image)
        loss = criterion(out, label)

        pred = nn.functional.softmax(out, dim=1)
        acc = accuracy(pred, label)

        batch_size = image.shape[0]
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        print(f"------ Validation, loss:{loss_meter.avg:.4}, acc@1: {acc_meter.avg:.2}")

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epoch, report_freq=10):
    print(f"--- Training Epoch [{epoch} / {total_epoch}]:")
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.train()
    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        device = get_device()
        image = image.to(device)
        label = label.to(device)

        out = model(image)
        loss = criterion(out, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred = nn.functional.softmax(out, dim=1)
        acc = accuracy(pred, label)

        batch_size = image.shape[0]
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        if batch_id > 0 and batch_id % report_freq == 0:
            print(f"------ Batch[{batch_id} / {len(dataloader)}, loss: {loss_meter.avg:.4}, Acc: {acc_meter.avg:.3}]")

    print(f"------ Epoch[{epoch} / {total_epoch}], loss: {loss_meter.avg:.4}, Acc:{acc_meter.avg:.3}")


def main():
    total_epochs = 160
    batch_size = 16
    device = get_device()
    model = ResNet18(num_classes=100)
    model = ViT(num_classes=100)
    model.to(device)
    # load state_dict
    state_dict = torch.load('./pths/vit/epoch2.pth')
    model.load_state_dict(state_dict)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=5)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=5, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    for epoch in range(3, total_epochs + 1):
        train_one_epoch(model, train_dataloader, criterion, optimizer, epoch, total_epochs)
        train_scheduler.step()
        validate(model, val_dataloader, criterion)
        torch.save(model.state_dict(), f'pths/vit/epoch{epoch}.pth')


if __name__ == "__main__":
    main()


