import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.sig = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.s4 = nn.AvgPool2d(2, stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(400, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.s2(x)
        x = self.sig(self.conv2(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))