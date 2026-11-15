"""PyTorch DL-M1 仿射基线：796参数，无激活函数。"""

from __future__ import annotations

import torch
import torch.nn as nn


class DL_M1(nn.Module):
    """输入 (B,1,H,W) → 输出 (B,4,H,W)。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(1, 4, kernel_size=1, padding=0, bias=True)
        self.conv7x7 = nn.Conv2d(4, 4, kernel_size=7, padding=3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        return self.conv7x7(x)
