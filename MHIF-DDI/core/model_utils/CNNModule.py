import torch.nn as nn

class CNNModule(nn.Module):
    def __init__(self,in_channels):
        super(CNNModule, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels*2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.in_channels*4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.in_channels*4, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(4, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = x.squeeze(2)

        return x