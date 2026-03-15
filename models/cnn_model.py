# ============================================================
# models/cnn_model.py — CNN Feature Extractor (ResNet50)
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models


class CNNFeatureExtractor(nn.Module):
    """
    Trích xuất đặc trưng ảnh viên thuốc bằng ResNet50 pretrained.
    Output: vector 2048 chiều cho mỗi ảnh.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Tải ResNet50 với trọng số pretrained ImageNet
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Bỏ lớp fully-connected cuối cùng → giữ lại feature map
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor ảnh [B, 3, 224, 224]
        Returns:
            features: Tensor [B, 2048]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten: [B, 2048, 1, 1] → [B, 2048]
        return x
