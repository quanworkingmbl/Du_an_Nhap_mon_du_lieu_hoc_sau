# ============================================================
# models/vit_model.py — ViT Feature Extractor (Vision Transformer)
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models


class ViTFeatureExtractor(nn.Module):
    """
    Trích xuất đặc trưng ảnh viên thuốc bằng ViT-B/16 pretrained.
    Output: vector 768 chiều cho mỗi ảnh.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = models.vit_b_16(weights=weights)

        # Bỏ head classifier → giữ lại feature extractor
        self.features = nn.Sequential(
            vit.conv_proj,       # Patch embedding
        )
        self.encoder = vit.encoder
        self.class_token = vit.class_token
        self.pos_embedding = vit.encoder.pos_embedding if hasattr(vit.encoder, 'pos_embedding') else None

        # Lưu lại toàn bộ ViT, chỉ bỏ head
        self.vit = vit
        self.vit.heads = nn.Identity()  # Bỏ classification head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor ảnh [B, 3, 224, 224]
        Returns:
            features: Tensor [B, 768]
        """
        return self.vit(x)  # [B, 768] — CLS token output
