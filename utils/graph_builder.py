# ============================================================
# utils/graph_builder.py — Xây dựng đồ thị từ feature vectors
# ============================================================

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_graph(features: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
    """
    Xây dựng đồ thị không có hướng từ ma trận đặc trưng.
    Hai node được nối cạnh nếu cosine similarity của chúng > threshold.

    Args:
        features  : Tensor [N, D] — N ảnh, mỗi ảnh có vector D chiều
        threshold : Ngưỡng similarity tối thiểu để nối cạnh (default 0.8)

    Returns:
        edge_index: Tensor [2, E] — danh sách cạnh định dạng PyG
    """
    # Chuyển về numpy để tính cosine similarity
    feats_np = features.detach().cpu().numpy()
    sim_matrix = cosine_similarity(feats_np)   # [N, N]

    edges = []
    n = len(sim_matrix)
    for i in range(n):
        for j in range(n):
            if i != j and sim_matrix[i][j] > threshold:
                edges.append([i, j])

    if len(edges) == 0:
        # Nếu không có cạnh nào → tạo self-loops để tránh lỗi
        print("[WARN] Không tìm thấy cạnh nào với threshold=%.2f. "
              "Thử giảm threshold. Đang dùng self-loops." % threshold)
        edges = [[i, i] for i in range(n)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"[Graph] Số node: {n} | Số cạnh: {edge_index.shape[1]} "
          f"| Threshold: {threshold}")
    return edge_index
