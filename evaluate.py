# ============================================================
# evaluate.py — Đánh giá model sau khi train
# ============================================================

import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from config import (
    CNN_FEATURE_DIM, GCN_HIDDEN_DIM,
    MODEL_SAVE_PATH, FEATURES_SAVE_PATH, LABELS_SAVE_PATH, EDGE_INDEX_SAVE,
    DATA_RAW_PATH,
)
from models.gcn_model import GCN
from utils.dataset_loader import load_dataset
from label_map import get_display_names

# ── Load class names ─────────────────────────────────────────
_, num_classes, class_names = load_dataset(DATA_RAW_PATH, batch_size=1, shuffle=False)
display_names = get_display_names(class_names)

# ── Load dữ liệu đã trích xuất ──────────────────────────────
print("[Load] Backbone: ViT-B/16")
print("[Load] Đọc features và labels từ file đã lưu...")
features   = torch.load(FEATURES_SAVE_PATH)
labels     = torch.load(LABELS_SAVE_PATH)
edge_index = torch.load(EDGE_INDEX_SAVE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features   = features.to(device)
labels     = labels.to(device)
edge_index = edge_index.to(device)

# ── Load model ───────────────────────────────────────────────
model = GCN(CNN_FEATURE_DIM, GCN_HIDDEN_DIM, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()
print(f"[Load] Model đã tải từ {MODEL_SAVE_PATH}")

# ── Dự đoán ──────────────────────────────────────────────────
with torch.no_grad():
    out  = model(features, edge_index)
    pred = out.argmax(dim=1)

# ── Các metric ───────────────────────────────────────────────
y_true = labels.cpu().numpy()
y_pred = pred.cpu().numpy()

accuracy = (y_pred == y_true).sum() / len(y_true)
print(f"\n{'='*50}")
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"{'='*50}")

print("\n[Classification Report]")
print(classification_report(y_true, y_pred, target_names=display_names, zero_division=0))

print("[Confusion Matrix]")
cm = confusion_matrix(y_true, y_pred)
print(np.array2string(cm, separator=" "))
print("\nLabel order:", display_names)
