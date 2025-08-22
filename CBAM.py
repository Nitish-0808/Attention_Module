import torch
import torch.nn as nn
## Channel Attention Module of CBAM
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False)
            )
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        avg_out=self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out=self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out=avg_out+max_out
        return self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)
## Spatial Attention Module of CBAM 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
       super().__init__()
       assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
       padding=3 if kernel_size==7 else 1
       self.conv1=nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
       self.sigmoid=nn.Sigmoid()
    def forward(self, x):
       avg_out=torch.mean(x, dim=1, keepdim=True)
       max_out, _=torch.max(x, dim=1, keepdim=True)
       x_concat=torch.cat([avg_out, max_out], dim=1)
       out=self.conv1(x_concat)
       return self.sigmoid(out)