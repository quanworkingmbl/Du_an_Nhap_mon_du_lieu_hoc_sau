# ============================================================
# utils/graph_builder.py — Xây dựng đồ thị từ feature vectors
# ============================================================

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_graph(features: torch.Tensor, threshold: float = 0.8, batch_size: int = 1000) -> torch.Tensor:
    """
    Xây dựng đồ thị không có hướng từ ma trận đặc trưng.
    Hai node được nối cạnh nếu cosine similarity > threshold.

    Hỗ trợ dataset lớn bằng cách tính similarity theo batch.

    Args:
        features   : Tensor [N, D]
        threshold  : Ngưỡng similarity tối thiểu (default 0.8)
        batch_size : Kích thước batch khi tính similarity (cho dataset lớn)

    Returns:
        edge_index : Tensor [2, E]
    """
    feats_np = features.detach().cpu().numpy()
    n = len(feats_np)

    edges = []

    if n <= batch_size:
        # Dataset nhỏ — tính trực tiếp
        sim_matrix = cosine_similarity(feats_np)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > threshold:
                    edges.append([i, j])
                    edges.append([j, i])
    else:
        # Dataset lớn — tính theo batch để tiết kiệm RAM
        print(f"[Graph] Dataset lớn ({n} nodes) — tính similarity theo batch...")
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sim_block = cosine_similarity(feats_np[start:end], feats_np)
            for i_local in range(end - start):
                i_global = start + i_local
                for j in range(n):
                    if i_global != j and sim_block[i_local][j] > threshold:
                        edges.append([i_global, j])
                        edges.append([j, i_global])
            if start % (batch_size * 5) == 0:
                print(f"  Processed {end}/{n} nodes...")

    if len(edges) == 0:
        print("[WARN] Không tìm thấy cạnh nào với threshold=%.2f. "
              "Thử giảm threshold. Đang dùng self-loops." % threshold)
        edges = [[i, i] for i in range(n)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"[Graph] Số node: {n} | Số cạnh: {edge_index.shape[1]} "
          f"| Threshold: {threshold}")
    return edge_index


def extend_graph_with_new_node(
    train_features: torch.Tensor,
    new_feat: torch.Tensor,
    existing_edge_index: torch.Tensor,
    threshold: float = 0.8,
    top_k: int = 5,
) -> torch.Tensor:
    """
    Mở rộng đồ thị hiện có bằng cách thêm node mới (ảnh cần nhận dạng)
    và kết nối nó với các node huấn luyện có cosine similarity > threshold.

    Nếu không tìm được cạnh nào, kết nối với top_k node gần nhất.

    Args:
        train_features       : Tensor [N, D] — features từ tập huấn luyện
        new_feat             : Tensor [1, D] — feature của ảnh mới
        existing_edge_index  : Tensor [2, E] — đồ thị gốc
        threshold            : Ngưỡng cosine similarity để tạo cạnh
        top_k                : Số cạnh tối thiểu nếu không đủ ngưỡng

    Returns:
        extended_edge_index  : Tensor [2, E + E_new] — đồ thị mở rộng
    """
    n = train_features.shape[0]
    new_idx = n  # Node mới nằm ở cuối

    train_np = train_features.detach().cpu().numpy()
    new_np = new_feat.detach().cpu().numpy()

    sim = cosine_similarity(new_np, train_np)[0]  # [N]

    new_edges = []
    for j in range(n):
        if sim[j] > threshold:
            new_edges.append([new_idx, j])
            new_edges.append([j, new_idx])

    # Dự phòng: kết nối với top_k node gần nhất khi không đủ ngưỡng
    if len(new_edges) == 0:
        top_indices = np.argpartition(sim, -min(top_k, n))[-min(top_k, n):]
        for j in top_indices:
            new_edges.append([new_idx, j])
            new_edges.append([j, new_idx])

    new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    new_edges_tensor = new_edges_tensor.to(existing_edge_index.device)
    return torch.cat([existing_edge_index, new_edges_tensor], dim=1)
