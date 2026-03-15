# ============================================================
# config.py — Cấu hình toàn bộ dự án ResNet + GCN
# ============================================================

import os

# ── Đường dẫn dataset ────────────────────────────────────────
# Sau khi chạy organize_dataset.py, ảnh sẽ được sắp xếp vào đây
DATA_RAW_PATH       = "dataset/organized/train"   # ImageFolder format
DATA_PROCESSED_PATH = "checkpoints"               # Nơi lưu features, labels, edge_index

# ── Đường dẫn lưu model & features ──────────────────────────
MODEL_SAVE_PATH    = os.path.join(DATA_PROCESSED_PATH, "best_model.pth")
FEATURES_SAVE_PATH = os.path.join(DATA_PROCESSED_PATH, "features.pt")
LABELS_SAVE_PATH   = os.path.join(DATA_PROCESSED_PATH, "labels.pt")
EDGE_INDEX_SAVE    = os.path.join(DATA_PROCESSED_PATH, "edge_index.pt")

# ── Tham số mô hình ──────────────────────────────────────────
CNN_INPUT_SIZE  = 224    # Kích thước ảnh đầu vào ResNet (224x224)
CNN_FEATURE_DIM = 2048   # Chiều output của ResNet50 (layer avgpool)
GCN_HIDDEN_DIM  = 256    # Số node ẩn trong GCN

# ── Tham số huấn luyện ───────────────────────────────────────
BATCH_SIZE   = 32
LEARNING_RATE = 0.001
NUM_EPOCHS   = 100

# ── Tham số xây dựng đồ thị ──────────────────────────────────
# Hai ảnh được nối cạnh nếu cosine similarity > ngưỡng này
SIMILARITY_THRESHOLD = 0.8
