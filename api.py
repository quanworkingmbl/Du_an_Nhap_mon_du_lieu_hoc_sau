# ============================================================
# api.py — Flask API cho UI phân tích AI (ViT + GCN)
# ============================================================

import os
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Biến toàn cục ─────────────────────────────────────────────
_model_loaded = False
_model = None
_features = None
_labels = None
_edge_index = None
_class_names = None
_num_classes = 0
_feature_extractor = None
_device = None


def _try_load_model():
    """Thử load model và dữ liệu từ checkpoints."""
    global _model_loaded, _model, _features, _labels, _edge_index
    global _class_names, _num_classes, _feature_extractor, _device

    if _model_loaded:
        return True

    try:
        import torch
        from config import (
            CNN_FEATURE_DIM, GCN_HIDDEN_DIM,
            MODEL_SAVE_PATH, FEATURES_SAVE_PATH,
            LABELS_SAVE_PATH, EDGE_INDEX_SAVE,
            DATA_RAW_PATH, SIMILARITY_THRESHOLD,
        )
        from models.gcn_model import GCN
        from models.vit_model import ViTFeatureExtractor

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load features, labels, edge_index
        _features = torch.load(FEATURES_SAVE_PATH, map_location=_device)
        _labels = torch.load(LABELS_SAVE_PATH, map_location=_device)
        _edge_index = torch.load(EDGE_INDEX_SAVE, map_location=_device)

        # Xác định class names
        try:
            from utils.dataset_loader import load_dataset
            _, _num_classes, _class_names = load_dataset(
                DATA_RAW_PATH, batch_size=1, shuffle=False
            )
            _class_names = list(_class_names)
        except Exception:
            unique_labels = torch.unique(_labels).cpu().tolist()
            _num_classes = len(unique_labels)
            _class_names = [f"Class_{i}" for i in range(_num_classes)]

        # Load GCN model
        _model = GCN(CNN_FEATURE_DIM, GCN_HIDDEN_DIM, _num_classes).to(_device)
        _model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=_device))
        _model.eval()

        # Load ViT cho inference
        _feature_extractor = ViTFeatureExtractor(pretrained=True).to(_device)
        _feature_extractor.eval()

        _model_loaded = True
        print("[API] Model loaded thành công! (Backbone: ViT-B/16)")
        return True

    except Exception as e:
        print(f"[API] Không thể load model: {e}")
        traceback.print_exc()
        return False


# ── Dữ liệu demo khi model chưa load ─────────────────────────

DEMO_CLASS_NAMES = [
    "Aspirin", "Ibuprofen", "Paracetamol",
    "Amoxicillin", "Vitamin_C",
]

DEMO_METRICS = {
    "accuracy": 0.946,
    "per_class": [
        {"name": "Aspirin", "precision": 0.95, "recall": 0.93, "f1": 0.94, "support": 120},
        {"name": "Ibuprofen", "precision": 0.92, "recall": 0.96, "f1": 0.94, "support": 105},
        {"name": "Paracetamol", "precision": 0.97, "recall": 0.94, "f1": 0.955, "support": 98},
        {"name": "Amoxicillin", "precision": 0.91, "recall": 0.95, "f1": 0.93, "support": 87},
        {"name": "Vitamin_C", "precision": 0.96, "recall": 0.95, "f1": 0.955, "support": 110},
    ],
    "confusion_matrix": [
        [112, 3, 2, 2, 1],
        [2, 101, 1, 0, 1],
        [1, 2, 92, 2, 1],
        [2, 1, 1, 83, 0],
        [1, 2, 1, 1, 105],
    ],
    "class_names": DEMO_CLASS_NAMES,
}


# ── API Endpoints ─────────────────────────────────────────────

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Trả về thông tin cấu hình model."""
    from config import (
        CNN_FEATURE_DIM, GCN_HIDDEN_DIM, CNN_INPUT_SIZE,
        BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, SIMILARITY_THRESHOLD,
    )
    from label_map import get_display_names

    model_loaded = _try_load_model()

    info = {
        "model_name": "ViT-B/16 + GCN (Graph Convolutional Network)",
        "model_loaded": model_loaded,
        "cnn": {
            "architecture": "ViT-B/16 (pretrained ImageNet)",
            "feature_dim": CNN_FEATURE_DIM,
            "input_size": CNN_INPUT_SIZE,
        },
        "gcn": {
            "hidden_dim": GCN_HIDDEN_DIM,
            "layers": 2,
            "dropout": 0.5,
        },
        "training": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
        },
    }

    if model_loaded:
        info["dataset"] = {
            "num_samples": int(_features.shape[0]),
            "num_classes": _num_classes,
            "class_names": get_display_names(_class_names),
            "feature_shape": list(_features.shape),
            "num_edges": int(_edge_index.shape[1]),
        }
    else:
        info["dataset"] = {
            "num_samples": 520,
            "num_classes": len(DEMO_CLASS_NAMES),
            "class_names": DEMO_CLASS_NAMES,
            "feature_shape": [520, CNN_FEATURE_DIM],
            "num_edges": 1840,
        }

    return jsonify(info)


@app.route("/api/evaluate", methods=["GET"])
def evaluate():
    """Trả về kết quả đánh giá model."""
    model_loaded = _try_load_model()

    if not model_loaded:
        return jsonify({"source": "demo", **DEMO_METRICS})

    try:
        import torch
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix

        with torch.no_grad():
            out = _model(_features, _edge_index)
            pred = out.argmax(dim=1)

        y_true = _labels.cpu().numpy()
        y_pred = pred.cpu().numpy()

        accuracy = float((y_pred == y_true).sum() / len(y_true))

        report = classification_report(
            y_true, y_pred,
            target_names=_class_names,
            output_dict=True,
            zero_division=0,
        )

        from label_map import get_display_name
        per_class = []
        for name in _class_names:
            if name in report:
                per_class.append({
                    "name": get_display_name(name),
                    "precision": round(report[name]["precision"], 4),
                    "recall": round(report[name]["recall"], 4),
                    "f1": round(report[name]["f1-score"], 4),
                    "support": int(report[name]["support"]),
                })

        from label_map import get_display_names as _get_display_names
        cm = confusion_matrix(y_true, y_pred).tolist()

        return jsonify({
            "source": "model",
            "accuracy": round(accuracy, 4),
            "per_class": per_class,
            "confusion_matrix": cm,
            "class_names": _get_display_names(_class_names),
        })

    except Exception as e:
        return jsonify({"error": str(e), "source": "demo", **DEMO_METRICS}), 200


@app.route("/api/inference", methods=["POST"])
def inference():
    """Nhận dạng ảnh viên thuốc."""
    if "image" not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh. Gửi file với key 'image'."}), 400

    model_loaded = _try_load_model()

    if not model_loaded:
        # Trả về kết quả demo
        import random
        probs = [random.random() for _ in DEMO_CLASS_NAMES]
        total = sum(probs)
        probs = [p / total for p in probs]
        best_idx = probs.index(max(probs))
        return jsonify({
            "source": "demo",
            "predicted_class": DEMO_CLASS_NAMES[best_idx],
            "confidence": round(max(probs), 4),
            "probabilities": [
                {"class": name, "probability": round(p, 4)}
                for name, p in zip(DEMO_CLASS_NAMES, probs)
            ],
        })

    try:
        import torch
        from PIL import Image
        from config import CNN_INPUT_SIZE, SIMILARITY_THRESHOLD
        from utils.dataset_loader import get_transforms
        from utils.graph_builder import extend_graph_with_new_node

        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")

        transform = get_transforms(CNN_INPUT_SIZE, augment=False)
        img_t = transform(img).unsqueeze(0).to(_device)

        with torch.no_grad():
            new_feat = _feature_extractor(img_t)
            combined_features = torch.cat([_features, new_feat], dim=0)
            extended_edge_index = extend_graph_with_new_node(
                _features, new_feat, _edge_index, threshold=SIMILARITY_THRESHOLD
            )
            out = _model(combined_features, extended_edge_index)
            last = out[-1].unsqueeze(0)
            probs = torch.softmax(last, dim=1)
            confidence = probs.max().item()
            pred_class = probs.argmax().item()

        from label_map import get_display_name
        probabilities = []
        for i, name in enumerate(_class_names):
            probabilities.append({
                "class": get_display_name(name),
                "probability": round(probs[0][i].item(), 4),
            })

        probabilities.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({
            "source": "model",
            "predicted_class": get_display_name(_class_names[pred_class]),
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Kiểm tra API hoạt động."""
    return jsonify({"status": "ok", "model_loaded": _model_loaded})


if __name__ == "__main__":
    print("[API] Khởi động Flask API trên cổng 5000...")
    print("[API] Truy cập: http://localhost:5000")
    _try_load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
