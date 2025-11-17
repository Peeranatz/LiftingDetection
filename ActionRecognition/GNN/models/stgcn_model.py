# Cell 5: (ฉบับแก้ไข) สร้างโมเดล ST-GCN (เพิ่มความจุ)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.graph_ntu import A_norm


# -----------------------------------------------------------
# Correct ST-GCN Implementation (Bas version fixed)
# -----------------------------------------------------------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 3, kernel_size=1)

    def forward(self, x, A):
        B, C, T, V = x.shape

        # 1x1 conv
        x = self.conv(x)  # (B,3*out,T,V)
        x = x.view(B, 3, -1, T, V)  # (B,3,C_out,T,V)

        # graph convolution with 3 partitions
        out = []
        for k in range(3):
            out_k = torch.einsum("bctv, vw -> bctw", x[:, k], A[k])
            out.append(out_k)

        return sum(out).contiguous()  # (B,C_out,T,V)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()

        self.gcn = GraphConv(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU(inplace=True)

        # correct residual
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.bn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class STGCN(nn.Module):
    def __init__(self, num_classes, in_channels=3, num_nodes=25):
        super().__init__()

        self.register_buffer("A", torch.tensor(A_norm, dtype=torch.float32))

        self.layers = nn.ModuleList(
            [
                STGCN_Block(in_channels, 64),
                STGCN_Block(64, 64),
                STGCN_Block(64, 128),
                STGCN_Block(128, 128),
                STGCN_Block(128, 256),
                STGCN_Block(256, 256),
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        A = self.A  # (3,25,25)
        for layer in self.layers:
            x = layer(x, A)
        x = self.pool(x).flatten(1)
        return self.fc(x)
