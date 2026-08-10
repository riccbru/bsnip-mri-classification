import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(Simple3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(32 * 16 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = Simple3DCNN()
    print(model)
    sample = torch.randn(1, 1, 128, 128, 128)
    out = model(sample)
    print(out.shape)  # Should be [1, 3] for 3 classes (CN, MCI, AD)
