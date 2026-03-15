# ============================================================
# inference.py — Nhận dạng một ảnh viên thuốc mới
# ============================================================

import sys
import torch
from PIL import Image

from config import (
    CNN_FEATURE_DIM, GCN_HIDDEN_DIM,
    MODEL_SAVE_PATH, FEATURES_SAVE_PATH, EDGE_INDEX_SAVE,
    DATA_RAW_PATH, CNN_INPUT_SIZE,
)
from models.cnn_model import CNNFeatureExtractor
from models.gcn_model import GCN
from utils.dataset_loader import get_transforms, load_dataset

# ── Đọc đường dẫn ảnh từ command line ───────────────────────
if len(sys.argv) < 2:
    print("Cách dùng: python inference.py <đường_dẫn_ảnh.jpg>")
    sys.exit(1)

img_path = sys.argv[1]

# ── Load class names ─────────────────────────────────────────
_, num_classes, class_names = load_dataset(DATA_RAW_PATH, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load CNN ──────────────────────────────────────────────────
cnn = CNNFeatureExtractor(pretrained=True).to(device)
cnn.eval()

# ── Load GCN ─────────────────────────────────────────────────
model = GCN(CNN_FEATURE_DIM, GCN_HIDDEN_DIM, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

# ── Load graph từ training ───────────────────────────────────
train_features = torch.load(FEATURES_SAVE_PATH).to(device)
edge_index     = torch.load(EDGE_INDEX_SAVE).to(device)

# ── Xử lý ảnh test ───────────────────────────────────────────
transform = get_transforms(CNN_INPUT_SIZE, augment=False)
img       = Image.open(img_path).convert("RGB")
img_t     = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

with torch.no_grad():
    # Trích xuất đặc trưng ảnh mới
    new_feat = cnn(img_t)                              # [1, 2048]

    # Ghép vào graph hiện có (node mới không có cạnh → dùng mean làm fake)
    # Cách đơn giản: dùng trực tiếp CNN feature qua GCN với graph cũ
    # Thêm node mới vào cuối features
    combined_features = torch.cat([train_features, new_feat], dim=0)   # [N+1, 2048]

    # Chạy GCN, lấy output của node cuối (node mới)
    out  = model(combined_features, edge_index)        # [N+1, num_classes]
    last = out[-1].unsqueeze(0)                        # [1, num_classes]

    probs      = torch.softmax(last, dim=1)
    confidence = probs.max().item()
    pred_class = probs.argmax().item()

# ── Kết quả ──────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"  Ảnh      : {img_path}")
print(f"  Predicted: {class_names[pred_class]}")
print(f"  Confidence: {confidence*100:.1f}%")
print(f"{'='*40}")
print("\nXác suất từng lớp:")
for i, name in enumerate(class_names):
    bar = "█" * int(probs[0][i].item() * 30)
    print(f"  {name:20s} {probs[0][i].item()*100:5.1f}%  {bar}")
