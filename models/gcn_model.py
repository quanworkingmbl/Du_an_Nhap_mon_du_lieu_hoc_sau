# ============================================================
# models/gcn_model.py — Graph Convolutional Network (GCN)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network để phân loại node (viên thuốc).

    Args:
        in_channels     : số chiều vector đặc trưng đầu vào (2048 từ CNN)
        hidden_channels : số chiều lớp ẩn (mặc định 256)
        out_channels    : số lớp phân loại (số loại thuốc)
        dropout         : tỉ lệ dropout giữa hai lớp GCN
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : Node feature matrix [N, in_channels]
            edge_index : Graph edge list [2, E]
        Returns:
            out        : Log-softmax class scores [N, out_channels]
        """
        # Lớp GCN 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Lớp GCN 2
        x = self.conv2(x, edge_index)

        return x    # CrossEntropyLoss tự áp softmax nên không cần thêm
