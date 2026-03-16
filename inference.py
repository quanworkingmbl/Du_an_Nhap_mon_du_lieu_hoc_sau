# ============================================================
# inference.py — Nhận dạng một ảnh viên thuốc mới
# ============================================================

import sys
import torch
from PIL import Image

from config import (
    CNN_FEATURE_DIM, GCN_HIDDEN_DIM,
    MODEL_SAVE_PATH, FEATURES_SAVE_PATH, EDGE_INDEX_SAVE,
    DATA_RAW_PATH, CNN_INPUT_SIZE, SIMILARITY_THRESHOLD,
)
from models.vit_model import ViTFeatureExtractor
from models.gcn_model import GCN
from utils.dataset_loader import get_transforms, load_dataset
from utils.graph_builder import extend_graph_with_new_node
from label_map import get_display_name

# ── Đọc đường dẫn ảnh từ command line ───────────────────────
if len(sys.argv) < 2:
    print("Cách dùng: python inference.py <đường_dẫn_ảnh.jpg>")
    sys.exit(1)

img_path = sys.argv[1]

# ── Load class names ─────────────────────────────────────────
_, num_classes, class_names = load_dataset(DATA_RAW_PATH, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load ViT ──────────────────────────────────────────────────
vit = ViTFeatureExtractor(pretrained=True).to(device)
vit.eval()

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
img_t     = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    new_feat = vit(img_t)
    combined_features = torch.cat([train_features, new_feat], dim=0)
    extended_edge_index = extend_graph_with_new_node(
        train_features, new_feat, edge_index, threshold=SIMILARITY_THRESHOLD
    )
    out  = model(combined_features, extended_edge_index)
    last = out[-1].unsqueeze(0)

    probs      = torch.softmax(last, dim=1)
    confidence = probs.max().item()
    pred_class = probs.argmax().item()

# ── Kết quả ──────────────────────────────────────────────────
display_name = get_display_name(class_names[pred_class])
print(f"\n{'='*40}")
print(f"  Ảnh       : {img_path}")
print(f"  Predicted : {display_name}")
print(f"  (Class ID : {class_names[pred_class]})")
print(f"  Confidence: {confidence*100:.1f}%")
print(f"{'='*40}")
print("\nXác suất từng lớp:")
for i, name in enumerate(class_names):
    disp = get_display_name(name)
    bar = "█" * int(probs[0][i].item() * 30)
    print(f"  {disp:30s} {probs[0][i].item()*100:5.1f}%  {bar}")
