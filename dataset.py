import torch
import torchvision
import torchvision.transforms as transforms

data_path = './dataset'

training_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
])
validation_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
])

train_dataset = torchvision.datasets.CIFAR100(root=data_path,
                                             train=True,
                                             transform=training_transform,
                                             download=True)
val_dataset = torchvision.datasets.CIFAR100(root=data_path,
                                           train=False,
                                           transform=validation_transforms,
                                           download=True)

