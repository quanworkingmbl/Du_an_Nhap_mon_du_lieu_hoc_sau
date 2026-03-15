# ============================================================
# config.py — Cấu hình toàn bộ dự án ViT + GCN
# ============================================================

import os

# ── Đường dẫn dataset ────────────────────────────────────────
DATA_RAW_PATH       = "dataset/organized/train"
DATA_PROCESSED_PATH = "checkpoints"

# ── Đường dẫn lưu model & features ──────────────────────────
MODEL_SAVE_PATH    = os.path.join(DATA_PROCESSED_PATH, "best_model.pth")
FEATURES_SAVE_PATH = os.path.join(DATA_PROCESSED_PATH, "features.pt")
LABELS_SAVE_PATH   = os.path.join(DATA_PROCESSED_PATH, "labels.pt")
EDGE_INDEX_SAVE    = os.path.join(DATA_PROCESSED_PATH, "edge_index.pt")

# ── Tham số mô hình ──────────────────────────────────────────
CNN_INPUT_SIZE  = 224         # Kích thước ảnh đầu vào (224x224)
CNN_FEATURE_DIM = 768         # ViT-B/16 output dimension
GCN_HIDDEN_DIM  = 256         # Số node ẩn trong GCN

# ── Tham số huấn luyện ───────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
NUM_EPOCHS    = 100

# ── Tham số xây dựng đồ thị ──────────────────────────────────
SIMILARITY_THRESHOLD = 0.8
