import os
import base64


def get_icon_path(file_name):
    """Mendapatkan path absolut agar folder 'icon' selalu ditemukan."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "icon", file_name)


def load_svg(file_name):
    file_path = get_icon_path(file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            f"<div style='width:24px; height:24px; border:1px dashed red;"
            f" font-size:10px; color:red; display:flex; align-items:center; justify-content:center;'"
            f" title='Missing: {file_name}'>X</div>"
        )


def get_svg_base64(file_name):
    file_path = get_icon_path(file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            b64 = base64.b64encode(f.read().encode("utf-8")).decode("utf-8")
            return f"data:image/svg+xml;base64,{b64}"
    except FileNotFoundError:
        return ""
