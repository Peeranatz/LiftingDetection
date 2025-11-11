# Cell 5: (ฉบับแก้ไข) สร้างโมเดล ST-GCN (เพิ่มความจุ)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. Graph Convolution (GCN)
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 3, kernel_size=1)
    def forward(self, x, A):
        B, C, T, V = x.shape
        x = self.conv(x); x = x.view(B, 3, -1, T, V)
        x = torch.einsum('bkctv, vw -> bkctw', (x, A))
        x = x[:, 0]; return x.contiguous()

# 2. Temporal Convolution (TCN)
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); return x

# 3. ST-GCN Block
class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(STGCN_Block, self).__init__()
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A); x = self.tcn(x)
        x = x + res; x = self.relu(x); return x

# 4. โมเดลหลัก STGCN (Main Model)
class STGCN(nn.Module):
    def __init__(self, num_classes, in_channels=3, num_nodes=25, A=None):
        super(STGCN, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes

        # ✅ ปรับตรงนี้ — ถ้า A ไม่มี ให้สร้าง I matrix
        if A is None:
            A = np.eye(num_nodes, dtype=np.float32)
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))

        # **** นี่คือจุดที่แก้ไขครับ ****
        # เราเปิด 2 Layer ที่เคยปิดไป กลับคืนมา
        self.layers = nn.ModuleList([
            STGCN_Block(in_channels, 64),
            STGCN_Block(64, 64),
            STGCN_Block(64, 128),
            STGCN_Block(128, 128),
            STGCN_Block(128, 256), # <--- เปิด
            STGCN_Block(256, 256)  # <--- เปิด
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes) # <--- แก้ FC กลับเป็น 256
        # **************************

    def forward(self, x):
        B, C, T, V = x.shape
        for layer in self.layers:
            x = layer(x, self.A)
     