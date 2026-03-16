# Dự Án Nhận Dạng Thuốc Bằng AI (ResNet50 + GCN)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![React](https://img.shields.io/badge/React-19.2.4-blue.svg)

## 📋 Mục Lục
- [Giới Thiệu](#giới-thiệu)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [Đánh Giá Khả Năng Train 100K Ảnh](#đánh-giá-khả-năng-train-100k-ảnh)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [API Endpoints](#api-endpoints)
- [Tối Ưu Hóa & Khuyến Nghị](#tối-ưu-hóa--khuyến-nghị)

---

## 🎯 Giới Thiệu

Dự án này là một hệ thống nhận dạng thuốc (viên nén/thuốc viên) sử dụng Deep Learning với kiến trúc kết hợp:
- **ResNet50 (CNN)**: Trích xuất đặc trưng hình ảnh (pretrained trên ImageNet)
- **Graph Convolutional Network (GCN)**: Phân loại dựa trên mối quan hệ giữa các đặc trưng

### Tính Năng Chính
✅ Trích xuất đặc trưng tự động bằng ResNet50
✅ Xây dựng đồ thị dựa trên độ tương đồng cosine
✅ Phân loại bằng GCN 2 lớp
✅ REST API với Flask
✅ Giao diện web React với 3 tab chính:
  - Dashboard: Tổng quan metrics và kiến trúc
  - Metrics Detail: Chi tiết per-class, confusion matrix
  - Inference: Nhận dạng ảnh real-time

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────┐
│   Ảnh Đầu   │
│  (224x224)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   ResNet50 (CNN)    │
│  Feature Extractor  │
│   [Frozen Weights]  │
└──────┬──────────────┘
       │
       ▼ Features [N, 2048]
┌─────────────────────┐
│   Graph Builder     │
│ Cosine Similarity   │
│  Threshold = 0.8    │
└──────┬──────────────┘
       │
       ▼ Graph (Nodes + Edges)
┌─────────────────────┐
│   GCN (2 Layers)    │
│  Layer 1: 2048→256  │
│  Layer 2: 256→Classes│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Predicted Class    │
│   + Probabilities   │
└─────────────────────┘
```

### Pipeline Training
1. **Feature Extraction**: CNN trích xuất features từ tất cả ảnh
2. **Graph Construction**: Xây dựng đồ thị từ similarity matrix
3. **GCN Training**: Train GCN trên đồ thị đã xây dựng
4. **Model Saving**: Lưu model + features + graph structure

---

## 💻 Yêu Cầu Hệ Thống

### Phần Cứng Tối Thiểu (Dataset Nhỏ < 5K Ảnh)
- **CPU**: Intel i5 hoặc tương đương
- **RAM**: 8 GB
- **Storage**: 5 GB trống
- **GPU**: Không bắt buộc (có thể train trên CPU)

### Phần Cứng Khuyến Nghị (Dataset Lớn 50K-100K Ảnh)
- **CPU**: Intel i7/i9 hoặc AMD Ryzen 7/9
- **RAM**: 32 GB trở lên
- **Storage**: 50 GB SSD
- **GPU**: NVIDIA RTX 3060/3070 trở lên (12GB+ VRAM)
- **CUDA**: 11.8 hoặc cao hơn

### Phần Mềm
- Python 3.8+
- Node.js 16+ (cho React frontend)
- pip (Python package manager)
- npm hoặc yarn

---

## 📦 Cài Đặt

### 1. Clone Repository
```bash
git clone https://github.com/quanworkingmbl/Du_an_Nhap_mon_du_lieu_hoc_sau.git
cd Du_an_Nhap_mon_du_lieu_hoc_sau
```

### 2. Cài Đặt Python Dependencies
```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

**Lưu ý**: Nếu có GPU, cài đặt PyTorch với CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Cài Đặt Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Sử Dụng

### Bước 1: Chuẩn Bị Dataset

Đặt ảnh theo cấu trúc ImageFolder:
```
dataset/organized/train/
├── Class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Class_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Hoặc sử dụng script tự động sắp xếp:
```bash
python organize_dataset.py --train_src dataset/image/train --train_dst dataset/organized/train
```

### Bước 2: Train Model
```bash
python train.py
```

**Output**:
- `checkpoints/best_model.pth`: Model weights đã train
- `checkpoints/features.pt`: CNN features của tất cả ảnh
- `checkpoints/labels.pt`: Labels tương ứng
- `checkpoints/edge_index.pt`: Cấu trúc đồ thị

**Thời gian train** (ước tính):
- 520 ảnh (demo): ~2-5 phút (CPU)
- 10K ảnh: ~30-60 phút (GPU), ~3-5 giờ (CPU)
- 100K ảnh: ~8-12 giờ (GPU), **KHÔNG khuyến nghị CPU**

### Bước 3: Đánh Giá Model
```bash
python evaluate.py
```

Output: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Bước 4: Chạy API Server
```bash
python api.py
```
Server chạy tại: `http://localhost:5000`

### Bước 5: Chạy Frontend
```bash
cd frontend
npm run dev
```
Frontend chạy tại: `http://localhost:5173`

### Bước 6: Inference Đơn Lẻ
```bash
python inference.py path/to/image.jpg
```

---

## ⚠️ Đánh Giá Khả Năng Train 100K Ảnh

### 🔴 **KẾT LUẬN: DỰ ÁN HIỆN TẠI CHƯA ỔN ĐỂ TRAIN 100K ẢNH**

Sau khi phân tích kỹ thuật, dự án **cần tối ưu hóa nghiêm trọng** trước khi train 100K ảnh. Dưới đây là các vấn đề chính:

---

### 🚨 **VẤN ĐỀ NGHIÊM TRỌNG**

#### 1. **Memory Bottleneck - Graph Builder** ⛔
**File**: `utils/graph_builder.py:24`

**Vấn đề**:
```python
sim_matrix = cosine_similarity(feats_np)   # [N, N]
```

- Với **100K ảnh**, similarity matrix có kích thước: `100,000 x 100,000 = 10 tỷ phần tử`
- **Memory cần**: `10,000,000,000 × 4 bytes (float32) = 40 GB` chỉ cho similarity matrix!
- Loop lồng nhau `O(N²)` để tạo edges (dòng 28-31) sẽ mất **hàng giờ**

**Tác động**:
- ❌ Hầu hết máy cá nhân không có đủ 40GB RAM
- ❌ Nếu có RAM, việc tính toán vẫn mất 2-4 giờ
- ❌ Swap memory sẽ làm hệ thống đơ cứng

#### 2. **GCN Training - Full Batch** ⛔
**File**: `train.py:79`

**Vấn đề**:
```python
out = model(features_d, edge_index_d)   # [N, num_classes]
```

- GCN train **toàn bộ graph** mỗi epoch (không có mini-batch)
- Với 100K nodes, mỗi forward pass cần:
  - Features: `100,000 × 2048 × 4 bytes = 819 MB`
  - Edges: Ước tính 5-10 triệu edges = `10,000,000 × 2 × 8 bytes = 160 MB`
  - GCN activations + gradients: ~2-3 GB

**Tổng VRAM/RAM cần**: ~4-5 GB/epoch

**Tác động**:
- ❌ GPU 8GB có thể không đủ
- ❌ Mỗi epoch mất 10-15 phút → 100 epochs = 16-25 giờ

#### 3. **Feature Extraction - Data Loading** ⚠️
**File**: `train.py:38-44`

**Vấn đề**:
- Load tuần tự từng batch, không parallel prefetching
- Không có caching cho features đã extract

**Tác động**:
- Feature extraction cho 100K ảnh: ~2-3 giờ (GPU), ~10-15 giờ (CPU)
- Lãng phí thời gian nếu train lại (không cache)

#### 4. **Storage Requirements** ⚠️
- Features: `100,000 × 2048 × 4 = 819 MB`
- Labels: `100,000 × 8 = 800 KB`
- Edge index: ~1-2 GB (tùy threshold)
- Model: ~10-20 MB

**Tổng**: ~3-4 GB artifacts

---

### 🔧 **KHUYẾN NGHỊ TỐI ƯU HÓA BẮT BUỘC**

#### **1. Graph Construction - Sử dụng KNN thay vì Full Similarity**
Thay vì tính `O(N²)` similarity matrix, chỉ kết nối K nearest neighbors:

**Cách làm**:
```python
# Thay thế trong utils/graph_builder.py
from sklearn.neighbors import NearestNeighbors

def build_graph_knn(features, k=10):
    """Chỉ kết nối k láng giềng gần nhất"""
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Bỏ chính nó
            edges.append([i, j])

    return torch.tensor(edges).t()
```

**Lợi ích**:
- Complexity: `O(N² log N)` → `O(N log N)` (dùng KD-tree/Ball-tree)
- Memory: `40 GB` → `~2 GB`
- Thời gian: `2-4 giờ` → `10-20 phút`

#### **2. GCN Training - Mini-Batch Sampling**
Sử dụng PyTorch Geometric `NeighborLoader` để train theo batch:

**Cách làm**:
```python
from torch_geometric.loader import NeighborLoader

data = Data(x=features, edge_index=edge_index, y=labels)
loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],  # 2-hop sampling
    batch_size=512,
    shuffle=True,
)

for batch in loader:
    out = model(batch.x, batch.edge_index)
    loss = loss_fn(out[:batch.batch_size], batch.y[:batch.batch_size])
```

**Lợi ích**:
- Giảm memory/epoch: `5 GB` → `~1 GB`
- Có thể train trên GPU 8GB
- Mỗi epoch nhanh hơn 3-5x

#### **3. Feature Caching & Parallel Loading**
```python
# Thêm vào config.py
NUM_WORKERS = 4  # Parallel data loading

# Sửa trong train.py
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                   num_workers=NUM_WORKERS, pin_memory=True)
```

**Lợi ích**:
- Giảm feature extraction time: `3 giờ` → `1.5 giờ`

#### **4. Mixed Precision Training (AMP)**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for epoch in range(NUM_EPOCHS):
    with autocast():
        out = model(features, edge_index)
        loss = loss_fn(out, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Lợi ích**:
- Giảm VRAM usage: ~40%
- Tăng tốc training: ~2x

---

### 📊 **DỰ ĐOÁN HIỆU NĂNG SAU TỐI ƯU HÓA**

| Metric | Hiện Tại (100K) | Sau Tối Ưu (100K) |
|--------|-----------------|-------------------|
| **Graph Building** | 2-4 giờ, 40GB RAM ❌ | 15-30 phút, 2GB RAM ✅ |
| **Feature Extraction** | 2-3 giờ (GPU) | 1-1.5 giờ (GPU) ✅ |
| **Training/Epoch** | 10-15 phút, 5GB VRAM ❌ | 2-3 phút, 1GB VRAM ✅ |
| **Total Training (100 epochs)** | 20-30 giờ ❌ | 4-6 giờ ✅ |
| **Min GPU** | RTX 3080 (10GB) | RTX 3060 (6GB) ✅ |
| **Min RAM** | 48 GB ❌ | 16 GB ✅ |

---

### ✅ **CHECKLIST TỐI ƯU HÓA**

Trước khi train 100K ảnh, BẮT BUỘC hoàn thành:

- [ ] **P0** (Bắt buộc): Thay thế `build_graph()` bằng KNN-based
- [ ] **P0** (Bắt buộc): Implement mini-batch GCN training với `NeighborLoader`
- [ ] **P1** (Cao): Thêm Mixed Precision Training (AMP)
- [ ] **P1** (Cao): Parallel data loading với `num_workers`
- [ ] **P2** (Trung bình): Thêm gradient checkpointing để giảm memory
- [ ] **P2** (Trung bình): Monitoring với tensorboard/wandb
- [ ] **P3** (Thấp): Thử distributed training (nếu có nhiều GPU)

---

### 🎓 **KẾT LUẬN**

**Hiện tại**: Dự án hoạt động tốt với **<5K ảnh**
**Khả năng**: Có thể scale lên **10K ảnh** (với máy mạnh)
**100K ảnh**: **CHƯA KHẢ THI** - Cần refactor nghiêm trọng

**Thời gian tối ưu hóa ước tính**: 2-3 ngày làm việc

**Khuyến nghị ngay**:
1. Test với 5K ảnh trước
2. Implement KNN graph builder
3. Test với 20K ảnh
4. Implement mini-batch GCN
5. Test với 50K ảnh
6. Fine-tune và scale lên 100K

---

## 📁 Cấu Trúc Dự Án

```
Du_an_Nhap_mon_du_lieu_hoc_sau/
├── README.md                      # Tài liệu này
├── requirements.txt               # Python dependencies
├── config.py                      # Cấu hình toàn bộ dự án
├── train.py                       # Pipeline training
├── evaluate.py                    # Đánh giá model
├── inference.py                   # Nhận dạng đơn lẻ
├── organize_dataset.py            # Sắp xếp dataset
├── api.py                         # Flask REST API
│
├── models/
│   ├── __init__.py
│   ├── cnn_model.py              # ResNet50 feature extractor
│   └── gcn_model.py              # GCN classifier (2 layers)
│
├── utils/
│   ├── __init__.py
│   ├── dataset_loader.py         # PyTorch DataLoader
│   └── graph_builder.py          # Graph construction
│
├── checkpoints/                   # Model artifacts
│   ├── best_model.pth            # Trained GCN weights
│   ├── features.pt               # CNN features
│   ├── labels.pt                 # Class labels
│   └── edge_index.pt             # Graph structure
│
├── dataset/                       # Dataset directory
│   └── organized/
│       └── train/
│           ├── Class_1/
│           ├── Class_2/
│           └── ...
│
└── frontend/                      # React UI
    ├── package.json
    ├── vite.config.js
    ├── src/
    │   ├── App.jsx               # Main app
    │   ├── main.jsx
    │   └── components/
    │       ├── Dashboard.jsx     # Overview metrics
    │       ├── MetricsDetail.jsx # Detailed analytics
    │       └── Inference.jsx     # Image recognition
    └── ...
```

---

## 🔌 API Endpoints

### Base URL: `http://localhost:5000`

#### 1. Health Check
```http
GET /api/health
```
**Response**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### 2. Model Info
```http
GET /api/model-info
```
**Response**:
```json
{
  "model_name": "ResNet50 + GCN",
  "model_loaded": true,
  "cnn": {
    "architecture": "ResNet50 (pretrained ImageNet)",
    "feature_dim": 2048,
    "input_size": 224
  },
  "gcn": {
    "hidden_dim": 256,
    "layers": 2,
    "dropout": 0.5
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "similarity_threshold": 0.8,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss"
  },
  "dataset": {
    "num_samples": 520,
    "num_classes": 5,
    "class_names": ["Aspirin", "Ibuprofen", "Paracetamol", "Amoxicillin", "Vitamin_C"],
    "num_edges": 1840
  }
}
```

#### 3. Evaluate Model
```http
GET /api/evaluate
```
**Response**:
```json
{
  "source": "model",
  "accuracy": 0.946,
  "per_class": [
    {
      "name": "Aspirin",
      "precision": 0.95,
      "recall": 0.93,
      "f1": 0.94,
      "support": 120
    }
  ],
  "confusion_matrix": [[112, 3, 2, 2, 1], ...],
  "class_names": ["Aspirin", "Ibuprofen", "Paracetamol", "Amoxicillin", "Vitamin_C"]
}
```

#### 4. Inference (Nhận Dạng Ảnh)
```http
POST /api/inference
Content-Type: multipart/form-data

image: <file>
```
**Response**:
```json
{
  "source": "model",
  "predicted_class": "Aspirin",
  "confidence": 0.9234,
  "probabilities": [
    {"class": "Aspirin", "probability": 0.9234},
    {"class": "Ibuprofen", "probability": 0.0432},
    {"class": "Paracetamol", "probability": 0.0234},
    {"class": "Amoxicillin", "probability": 0.0078},
    {"class": "Vitamin_C", "probability": 0.0022}
  ]
}
```

---

## 🎨 Giao Diện Web

### Dashboard Tab
- Hiển thị accuracy tổng quát
- Visualize kiến trúc model
- Bảng thông số training
- Biểu đồ F1-Score per class
- Pie chart phân phối samples

### Metrics Detail Tab
- So sánh Precision/Recall/F1 chi tiết
- Highlight best/worst performing class
- Radar chart đa chiều
- Confusion matrix heatmap
- Bảng metrics đầy đủ

### Inference Tab
- Drag & drop upload ảnh
- Preview ảnh real-time
- Kết quả nhận dạng với confidence
- Bar chart xác suất tất cả classes
- Color-coded confidence (green/yellow/red)

---

## 🔧 Tối Ưu Hóa & Khuyến Nghị

### Cho Dataset Nhỏ (<10K Ảnh)
1. Giữ nguyên cấu hình hiện tại
2. Có thể train trên CPU (chậm nhưng khả thi)
3. Điều chỉnh `SIMILARITY_THRESHOLD` nếu graph quá thưa

### Cho Dataset Trung Bình (10K-50K Ảnh)
1. **Bắt buộc GPU**: Khuyến nghị GTX 1660 trở lên
2. Giảm `BATCH_SIZE` xuống 16 nếu hết VRAM
3. Tăng `NUM_WORKERS` trong DataLoader
4. Xem xét giảm `CNN_FEATURE_DIM` xuống 1024 (dùng ResNet34)

### Cho Dataset Lớn (50K-100K Ảnh) - YÊU CẦU TỐI ƯU HÓA
**Xem phần [Đánh Giá Khả Năng Train 100K Ảnh](#đánh-giá-khả-năng-train-100k-ảnh)**

---

## 🐛 Troubleshooting

### 1. Out of Memory (OOM)
**Lỗi**: `CUDA out of memory` hoặc `Killed` (RAM đầy)

**Giải pháp**:
```python
# Trong config.py
BATCH_SIZE = 16  # Giảm từ 32
GCN_HIDDEN_DIM = 128  # Giảm từ 256
```

### 2. Graph Quá Thưa (Ít Cạnh)
**Lỗi**: `[WARN] Không tìm thấy cạnh nào`

**Giải pháp**:
```python
# Trong config.py
SIMILARITY_THRESHOLD = 0.6  # Giảm từ 0.8
```

### 3. Training Quá Chậm
**Giải pháp**:
- Kiểm tra GPU: `nvidia-smi`
- Cài PyTorch với CUDA
- Giảm `NUM_EPOCHS` xuống 50 để test nhanh

### 4. Frontend Không Kết Nối API
**Giải pháp**:
```bash
# Kiểm tra API đang chạy
curl http://localhost:5000/api/health

# Kiểm tra CORS
# Đảm bảo flask-cors đã cài đặt
pip install flask-cors
```

---

## 📚 Dependencies Chính

### Backend
- **PyTorch**: Deep learning framework
- **torch-geometric**: Graph neural networks
- **torchvision**: Computer vision (ResNet50)
- **scikit-learn**: Metrics và similarity computation
- **Flask**: REST API server
- **Pillow**: Image processing

### Frontend
- **React**: UI framework
- **Vite**: Build tool & dev server
- **Recharts**: Charting library
- **React DOM**: React rendering

---

## 👥 Đóng Góp

Dự án này là project học tập. Mọi đóng góp đều được chào đón!

### Các Cải Tiến Đề Xuất
1. Implement KNN-based graph construction (P0)
2. Mini-batch GCN training (P0)
3. Data augmentation nâng cao
4. Thêm attention mechanism vào GCN
5. Web scraping để tự động thu thập dataset
6. Mobile app cho inference on-device
7. Docker containerization
8. CI/CD pipeline

---

## 📄 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

## 📞 Liên Hệ

- **Repository**: https://github.com/quanworkingmbl/Du_an_Nhap_mon_du_lieu_hoc_sau
- **Issues**: https://github.com/quanworkingmbl/Du_an_Nhap_mon_du_lieu_hoc_sau/issues

---

## 🎓 Tài Liệu Tham Khảo

1. **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. **GCN**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
3. **PyTorch Geometric**: [Official Documentation](https://pytorch-geometric.readthedocs.io/)

---

**Last Updated**: March 2026
**Project Status**: ✅ Hoạt động tốt với <5K ảnh | ⚠️ Cần tối ưu hóa cho 100K ảnh
