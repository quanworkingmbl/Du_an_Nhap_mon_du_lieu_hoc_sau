"""
organize_dataset.py
===================
Script tổ chức ảnh flat → ImageFolder format.

Ảnh hiện tại nằm flat trong:
  dataset/image/train/   (tên file: NIH-Pill-rximage_A_0002.jpg, Personal-251124_I0001.jpg, ...)
  dataset/image/val/

Script này sẽ phân tích tên file → tạo subfolder class → copy ảnh vào.

Cách chạy:
    python organize_dataset.py
    python organize_dataset.py --dry_run   ← chỉ xem, không di chuyển
"""

import os
import shutil
import argparse
import re
from collections import defaultdict
from pathlib import Path


def extract_class(filename: str) -> str:
    """
    Trích xuất tên class từ tên file.

    Quy tắc (dựa trên dataset NLM Pills):
        NIH-Pill-rximage_A_0002.jpg    → NIH_rximage_A
        NIH-Pill-rximage_m_0001.jpg    → NIH_rximage_m
        NIH-Pill-rximage_mosaic_0001.jpg → NIH_rximage_mosaic
        Personal-251124_I0001.jpg      → Personal_I
        Personal-251127_J0001.jpg      → Personal_J
        Personal-251127_K0001.jpg      → Personal_K
        Negatives_I0001.jpg            → Negatives
        ...
    """
    name = Path(filename).stem  # bỏ .jpg

    # NIH-Pill-rximage_<letter>_<num>
    m = re.match(r'NIH-Pill-rximage_([A-Za-z]+)_\d+', name)
    if m:
        return f"NIH_{m.group(1)}"

    # Personal-<date>_<letter><num>
    m = re.match(r'Personal-\d+_([A-Z])\d+', name)
    if m:
        return f"Personal_{m.group(1)}"

    # Personal-<date>_IMG_<num>
    m = re.match(r'Personal-\d+_IMG_\d+', name)
    if m:
        return "Personal_IMG"

    # Negatives_*
    if name.startswith('Negatives_'):
        return 'Negatives'

    # fallback: lấy phần trước dấu _
    return name.split('_')[0]


def organize_folder(src_dir: str, dst_dir: str, dry_run: bool = False):
    """
    Di chuyển ảnh từ src_dir (flat) → dst_dir/<class>/<img>.

    Nếu src_dir == dst_dir: tạo subfolder trong cùng thư mục (in-place).
    """
    src  = Path(src_dir)
    dst  = Path(dst_dir)
    imgs = list(src.glob("*.jpg")) + list(src.glob("*.png")) + list(src.glob("*.jpeg"))

    if not imgs:
        print(f"[!] Không tìm thấy ảnh trong: {src}")
        return

    stats = defaultdict(int)
    for img_path in imgs:
        cls  = extract_class(img_path.name)
        dest = dst / cls / img_path.name

        stats[cls] += 1

        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if img_path.resolve() != dest.resolve():
                shutil.copy2(str(img_path), str(dest))  # copy → không xoá gốc

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Kết quả phân loại: {src}")
    print(f"  Tổng ảnh   : {len(imgs)}")
    print(f"  Số class   : {len(stats)}")
    for cls, count in sorted(stats.items()):
        print(f"    {cls:<30} {count:>4} ảnh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_src", default="dataset/image/train",
                    help="Thư mục train gốc (flat images)")
    ap.add_argument("--val_src",   default="dataset/image/val",
                    help="Thư mục val gốc (flat images)")
    ap.add_argument("--train_dst", default="dataset/organized/train",
                    help="Thư mục train đích (ImageFolder format)")
    ap.add_argument("--val_dst",   default="dataset/organized/val",
                    help="Thư mục val đích (ImageFolder format)")
    ap.add_argument("--dry_run",   action="store_true",
                    help="Chỉ xem kết quả phân loại, không copy file")
    args = ap.parse_args()

    print("=" * 60)
    print("Organize Dataset → ImageFolder format")
    print("=" * 60)

    organize_folder(args.train_src, args.train_dst, args.dry_run)
    organize_folder(args.val_src,   args.val_dst,   args.dry_run)

    if not args.dry_run:
        print(f"\n✅ Xong! Dataset đã được sắp xếp vào:")
        print(f"   Train: {args.train_dst}")
        print(f"   Val  : {args.val_dst}")
        print(f"\nSau đó cập nhật config.py:")
        print(f"   DATA_RAW_PATH = \"{args.train_dst}\"")
    else:
        print("\n[DRY RUN] Không có file nào bị di chuyển.")
        print("Chạy không có --dry_run để thực sự copy file.")


if __name__ == "__main__":
    main()
