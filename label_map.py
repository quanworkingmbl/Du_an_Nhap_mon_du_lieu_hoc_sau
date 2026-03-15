# ============================================================
# label_map.py — Bảng ánh xạ mã class → tên thuốc thật
# ============================================================
#
# Cập nhật tên thuốc thật vào đây nếu biết.
# Format:  "tên_folder_class": "Tên thuốc hiển thị"
#

LABEL_MAP = {
    "pills_11":  "Viên tròn trắng (Nhóm 11)",
    "pills_23":  "Viên bầu dục (Nhóm 23)",
    "pills_32":  "Viên nang (Nhóm 32)",
    "pills_34":  "Viên dẹt (Nhóm 34)",
    "pills_43":  "Viên nang mềm (Nhóm 43)",
    "pills_ff":  "Viên hình oval (Nhóm FF)",
    "pills_t2":  "Viên 2 lớp (Type 2)",
    "pills_t2r": "Viên 2 lớp tròn (Type 2R)",
    "pills_t4":  "Viên 4 cạnh (Type 4)",
    "pills_t4r": "Viên 4 cạnh tròn (Type 4R)",
    "pills_t6":  "Viên lục giác (Type 6)",
    "pills_t6r": "Viên lục giác tròn (Type 6R)",
    "pills_t9":  "Viên đặc biệt (Type 9)",
    "pills_t9r": "Viên đặc biệt tròn (Type 9R)",
    "pills_IMG": "Viên chụp thực tế",
    "pills_ya4": "Viên thuốc ya4",
    "Various":   "Hỗn hợp nhiều loại",
}


def get_display_name(class_name: str) -> str:
    """Trả về tên hiển thị cho class. Nếu không có trong map → trả về tên gốc."""
    return LABEL_MAP.get(class_name, class_name)


def get_display_names(class_names: list) -> list:
    """Chuyển đổi danh sách tên class → tên hiển thị."""
    return [get_display_name(name) for name in class_names]
