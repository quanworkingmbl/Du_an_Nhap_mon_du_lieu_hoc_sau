# ============================================================
# train.py — Pipeline: ViT feature extraction → Graph → GCN training
# ============================================================

import os
import torch
import torch.nn as nn

from config import (
    DATA_RAW_PATH, DATA_PROCESSED_PATH,
    CNN_FEATURE_DIM, GCN_HIDDEN_DIM,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    SIMILARITY_THRESHOLD,
    MODEL_SAVE_PATH, FEATURES_SAVE_PATH, LABELS_SAVE_PATH, EDGE_INDEX_SAVE,
)
from models.vit_model import ViTFeatureExtractor
from models.gcn_model import GCN
from utils.dataset_loader import load_dataset
from utils.graph_builder import build_graph
from label_map import get_display_names

# ── Device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Sử dụng: {device}")

# ── 1. Load dataset ─────────────────────────────────────────
loader, num_classes, class_names = load_dataset(
    DATA_RAW_PATH, batch_size=BATCH_SIZE, augment=True
)

# ── 2. ViT Feature Extraction ───────────────────────────────
print("\n[Step 1] Trích xuất đặc trưng bằng ViT-B/16...")
vit = ViTFeatureExtractor(pretrained=True).to(device)
vit.eval()

all_features = []
all_labels   = []

with torch.no_grad():
    for batch_idx, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(device)
        feats = vit(imgs)
        all_features.append(feats.cpu())
        all_labels.append(lbls)
        print(f"  Batch {batch_idx+1}/{len(loader)} — {feats.shape}")

features = torch.cat(all_features)   # [N, 768]
labels   = torch.cat(all_labels)     # [N]
print(f"[ViT] Tổng số ảnh: {features.shape[0]} | Feature dim: {features.shape[1]}")

# ── 3. Lưu features để tái sử dụng ──────────────────────────
os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
torch.save(features,  FEATURES_SAVE_PATH)
torch.save(labels,    LABELS_SAVE_PATH)
print(f"[Saved] Features → {FEATURES_SAVE_PATH}")

# ── 4. Xây dựng đồ thị ──────────────────────────────────────
print("\n[Step 2] Xây dựng đồ thị (cosine similarity)...")
edge_index = build_graph(features, threshold=SIMILARITY_THRESHOLD)
torch.save(edge_index, EDGE_INDEX_SAVE)
print(f"[Saved] Edge index → {EDGE_INDEX_SAVE}")

# ── 5. Khởi tạo GCN ─────────────────────────────────────────
print("\n[Step 3] Train GCN...")
model     = GCN(CNN_FEATURE_DIM, GCN_HIDDEN_DIM, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn   = nn.CrossEntropyLoss()

# Chuyển dữ liệu lên device
features_d   = features.to(device)
labels_d     = labels.to(device)
edge_index_d = edge_index.to(device)

# ── Training loop ────────────────────────────────────────────
best_loss = float("inf")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    out  = model(features_d, edge_index_d)   # [N, num_classes]
    loss = loss_fn(out, labels_d)

    loss.backward()
    optimizer.step()

    # Tính accuracy nhanh
    pred     = out.argmax(dim=1)
    accuracy = (pred == labels_d).float().mean().item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | Loss: {loss.item():.4f} | Acc: {accuracy*100:.1f}%")

    # Lưu model tốt nhất
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

display_names = get_display_names(class_names)
print(f"\n[Done] Model đã lưu → {MODEL_SAVE_PATH}")
print(f"       Backbone : ViT-B/16")
print(f"       Best loss: {best_loss:.4f}")
print(f"       Số lớp   : {num_classes} → {display_names}")
