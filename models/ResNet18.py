import torch.nn as nn
import torch

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
       return x

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)

        if in_dim != out_dim or stride == 2:
            self.downsample = nn.Sequential(*[
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_dim)
            ])
        else:
            self.downsample = Identity()

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(h)
        x = x + identity
        x = self.relu(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, in_dim=64, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=in_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU()

        self.in_dim = in_dim
        self.layer1 = self._make_layers(dim=64, n_blocks=2, stride=1)
        self.layer2 = self._make_layers(dim=128, n_blocks=2, stride=2)
        self.layer3 = self._make_layers(dim=256, n_blocks=2, stride=2)
        self.layer4 = self._make_layers(dim=512, n_blocks=2, stride=2)

        # head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, dim, n_blocks, stride):
        layer_list = []
        layer_list.append(Block(self.in_dim, dim, stride=stride))
        self.in_dim = dim
        for i in range(1, n_blocks):
            layer_list.append(Block(self.in_dim, dim, stride=1))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

def main():
    x = torch.randn((4, 3, 32, 32))
    model = ResNet18()
    y = model(x)
    print(y.shape)

if __name__ == "__main__":
    main()
