import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes):
        super(self, Model).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.pool = nn.MaxPool2d(stride=(2, 2), kernel_size=2)

        self.fc1 = nn.Linear(64 * 111 * 111, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.drop_out = nn.Dropout(p=0.5)

    def forward(self, x):
        x = nn.ReLU(self.conv1(x))
        x = self.pool(x)
        x = nn.ReLU(self.conv2(x))
        x = self.pool(x)
        x = nn.ReLU(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = nn.ReLU(self.fc1(x))
        x = self.drop_out(x)
        x = self.fc2(x)

        return x