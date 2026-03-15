# ============================================================
# utils/dataset_loader.py — Load dataset ảnh viên thuốc
# ============================================================

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_transforms(input_size: int = 224, augment: bool = False):
    """
    Trả về transform pipeline.
    augment=True: dùng augmentation mạnh hơn cho training ViT.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225],    # ImageNet std
    )

    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])


def load_dataset(
    path: str,
    batch_size: int = 32,
    input_size: int = 224,
    augment: bool = False,
    shuffle: bool = True,
):
    """
    Load dataset từ thư mục theo cấu trúc ImageFolder.

    Cấu trúc thư mục:
        path/
          aspirin/  img1.jpg  img2.jpg ...
          ibuprofen/ img3.jpg ...

    Args:
        path       : Đường dẫn thư mục gốc chứa các class-folder
        batch_size : Số ảnh mỗi batch
        input_size : Kích thước resize ảnh (default 224 cho ResNet)
        augment    : Có dùng data augmentation không
        shuffle    : Shuffle dữ liệu hay không

    Returns:
        loader     : DataLoader
        num_classes: Số lớp phân loại
        class_names: Danh sách tên lớp
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {path}\n"
            "Hãy đặt ảnh vào data/raw/<tên_thuốc>/<ảnh>.jpg"
        )

    transform = get_transforms(input_size, augment)
    dataset   = ImageFolder(path, transform=transform)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    print(f"[Dataset] Đường dẫn: {path}")
    print(f"[Dataset] Tổng ảnh : {len(dataset)}")
    print(f"[Dataset] Số lớp   : {len(dataset.classes)} → {dataset.classes}")

    return loader, len(dataset.classes), dataset.classes
