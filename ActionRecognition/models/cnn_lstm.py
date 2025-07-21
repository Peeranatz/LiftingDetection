import torch.nn as nn
from torchvision import models

class CNNLSTMActionModel(nn.Module):
    def __init__(self, hidden_size=256, num_classes=4):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(base.children())[:-1])  # remove FC → output: [B, 512, 1, 1]
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x).squeeze()       # [B*T, 512]
        feats = feats.view(B, T, -1)        # [B, T, 512]
        _, (h_n, _) = self.lstm(feats)      # h_n: [1, B, H]
        out = self.fc(h_n.squeeze(0))       # [B, num_classes]
        return out