import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.ResNet18 import ResNet18
from dataset import train_dataset
from dataset import val_dataset
from torch.utils.data import DataLoader
# 这里如果有报错，无需理会，直接忽略即可
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """
    warmup_training learning rate scheduler
    Args:
        optimizer: optimizer(e.g. SGD)
        total_iters: total_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        we will use the first m batches, and set the learning rate to base_lr * m / total_iters
        :return:
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    topk的意思是模型预测的前k个中是否包含了正确答案
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def train(trainloader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    for i, (input, target) in enumerate(trainloader):
        input, target = input.cuda(), target.cuda()

        output = model(input)

        loss = criterion(output, target)

        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 100 - top1.avg

def validate(val_dataloader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_dataloader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(loss.item(), input.size(0))
    return 100 - top1.avg


# 超参数
warm=1
epoch=160
batch_size=128
loss_function = nn.CrossEntropyLoss()
net = ResNet18().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
trainloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=5, pin_memory=True)
valloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=5, pin_memory=True)
iter_per_epoch = len(trainloader)
warmip_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

def main():
    best_prec = 0
    for e in range(epoch):
        train_scheduler.step(e)
        train_acc = train(trainloader, net, loss_function, optimizer, e)
        test_acc = validate(valloader, net, loss_function)
        print(f'epoch: {e} train_acc: {train_acc:.3} test_acc: {test_acc:.3}')
        best_prec = max(test_acc, best_prec)
    print(f'The best prec is : {best_prec}')

if __name__ == '__main__':
    main()