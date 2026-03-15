"""
augment_dataset.py
==================
Script tăng cường dữ liệu (data augmentation) để đạt ~10,000 ảnh
cho việc huấn luyện ViT.

Cách chạy:
    python augment_dataset.py
    python augment_dataset.py --target 10000
    python augment_dataset.py --src dataset/organized/train --target 10000
    python augment_dataset.py --dry_run   ← chỉ xem, không tạo file
"""

import os
import math
import random
import argparse
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance


# ── Các phép augmentation ────────────────────────────────────────────────────

def augment_image(img: Image.Image, seed: int = None) -> Image.Image:
    """
    Áp dụng ngẫu nhiên một số phép biến đổi ảnh để tạo ảnh mới.

    Các phép biến đổi:
        - Xoay ngẫu nhiên (±30°)
        - Lật ngang / lật dọc
        - Thay đổi độ sáng, tương phản, độ bão hòa, độ sắc nét
        - Làm mờ Gaussian
        - Cắt ngẫu nhiên và phóng to lại
    """
    if seed is not None:
        random.seed(seed)

    # Xoay ngẫu nhiên
    if random.random() > 0.3:
        angle = random.uniform(-30, 30)
        img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))

    # Lật ngang
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Lật dọc
    if random.random() > 0.7:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Độ sáng
    if random.random() > 0.3:
        factor = random.uniform(0.6, 1.4)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Tương phản
    if random.random() > 0.3:
        factor = random.uniform(0.6, 1.4)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Độ bão hòa màu
    if random.random() > 0.4:
        factor = random.uniform(0.5, 1.5)
        img = ImageEnhance.Color(img).enhance(factor)

    # Độ sắc nét
    if random.random() > 0.5:
        factor = random.uniform(0.5, 2.0)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    # Làm mờ Gaussian nhẹ
    if random.random() > 0.6:
        radius = random.uniform(0.3, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Cắt ngẫu nhiên rồi phóng to lại kích thước ban đầu
    if random.random() > 0.4:
        w, h = img.size
        crop_factor = random.uniform(0.75, 0.95)
        new_w = int(w * crop_factor)
        new_h = int(h * crop_factor)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.BILINEAR)

    return img


# ── Hàm augment từng folder class ───────────────────────────────────────────

def augment_class_folder(
    cls_dir: Path,
    target_per_class: int,
    dry_run: bool = False,
) -> int:
    """
    Tạo thêm ảnh augmented cho một class folder cho đến khi đạt target_per_class.

    Args:
        cls_dir          : Đường dẫn thư mục class (vd: dataset/organized/train/pills_11)
        target_per_class : Số ảnh mục tiêu cho class này
        dry_run          : Nếu True, chỉ in thông tin, không tạo file

    Returns:
        Số ảnh đã được tạo thêm
    """
    imgs = (
        list(cls_dir.glob("*.jpg"))
        + list(cls_dir.glob("*.jpeg"))
        + list(cls_dir.glob("*.png"))
    )
    # Lọc bỏ ảnh augmented đã tạo trước đó (bắt đầu bằng "aug_")
    orig_imgs = [p for p in imgs if not p.name.startswith("aug_")]
    current   = len(imgs)
    needed    = max(0, target_per_class - current)

    print(f"    Hiện có : {current:>5} ảnh  |  Ảnh gốc: {len(orig_imgs)}  |  Cần thêm: {needed}")

    if needed == 0:
        print(f"    ✅ Đã đủ số lượng ảnh.")
        return 0

    if dry_run:
        print(f"    [DRY RUN] Sẽ tạo {needed} ảnh augmented.")
        return needed

    if not orig_imgs:
        print(f"    [!] Không có ảnh gốc để augment.")
        return 0

    generated = 0
    idx       = 0
    while generated < needed:
        src_path = orig_imgs[idx % len(orig_imgs)]
        idx += 1

        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"    [!] Lỗi đọc ảnh {src_path.name}: {e}")
            continue

        aug_img = augment_image(img, seed=generated * 1000 + idx)

        aug_name = f"aug_{generated:05d}_{src_path.stem}.jpg"
        aug_path = cls_dir / aug_name
        aug_img.save(aug_path, "JPEG", quality=95)
        generated += 1

        if generated % 100 == 0:
            print(f"      → Đã tạo {generated}/{needed} ảnh...")

    print(f"    ✅ Đã tạo thêm {generated} ảnh augmented.")
    return generated


# ── Hàm tổng — augment toàn bộ dataset ──────────────────────────────────────

def augment_dataset(src_dir: str, target_total: int, dry_run: bool = False):
    """
    Tăng cường toàn bộ dataset đến target_total ảnh.

    Args:
        src_dir      : Thư mục gốc chứa các subfolder class
                       (vd: dataset/organized/train)
        target_total : Tổng số ảnh mục tiêu (mặc định 10000)
        dry_run      : Nếu True, chỉ in thông tin, không tạo file
    """
    src = Path(src_dir)
    if not src.exists():
        print(f"[!] Thư mục không tồn tại: {src}")
        print("    Hãy chạy organize_dataset.py trước để tổ chức ảnh.")
        return

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"[!] Không tìm thấy thư mục class trong: {src}")
        print("    Cấu trúc cần có: <src>/<tên_class>/<ảnh>.jpg")
        return

    num_classes      = len(class_dirs)
    target_per_class = math.ceil(target_total / num_classes)

    # Đếm tổng ảnh hiện có
    current_total = sum(
        len(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png")))
        for d in class_dirs
    )

    print("=" * 60)
    print("Augment Dataset → ViT Training")
    print("=" * 60)
    print(f"  Thư mục nguồn  : {src}")
    print(f"  Số class       : {num_classes}")
    print(f"  Ảnh hiện có    : {current_total}")
    print(f"  Mục tiêu tổng  : {target_total}")
    print(f"  Mục tiêu/class : {target_per_class}")
    if dry_run:
        print("  [DRY RUN] Không tạo file thực sự.")
    print()

    total_generated = 0
    for cls_dir in class_dirs:
        print(f"  [Class] {cls_dir.name}")
        generated = augment_class_folder(cls_dir, target_per_class, dry_run)
        total_generated += generated

    # Tổng kết
    final_total = sum(
        len(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png")))
        for d in class_dirs
    )

    print()
    print("=" * 60)
    if dry_run:
        print(f"[DRY RUN] Sẽ tạo thêm ~{total_generated} ảnh.")
        print(f"          Tổng sau augment: ~{current_total + total_generated} ảnh")
        print("\nChạy không có --dry_run để thực sự tạo ảnh.")
    else:
        print(f"✅ Hoàn tất! Đã tạo thêm {total_generated} ảnh.")
        print(f"   Tổng ảnh trong dataset: {final_total}")
        print(f"\nBước tiếp theo:")
        print(f"   python train.py")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Tăng cường dữ liệu ảnh để đạt mục tiêu số lượng cho ViT training."
    )
    ap.add_argument(
        "--src",
        default="dataset/organized/train",
        help="Thư mục gốc chứa class subfolders (default: dataset/organized/train)",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=10000,
        help="Tổng số ảnh mục tiêu (default: 10000)",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Chỉ xem ước tính, không tạo file",
    )
    args = ap.parse_args()

    augment_dataset(args.src, args.target, args.dry_run)


if __name__ == "__main__":
    main()
