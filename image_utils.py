from typing import Optional
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image


def load_thumbnail_pixmap(path: str, w: int, h: int) -> Optional[QPixmap]:
    """
    Load a thumbnail with aspect-ratio preserved and smooth transform.
    Falls back to PIL+QImage if QPixmap fails.
    """
    pix = QPixmap(path)
    if pix.isNull():
        try:
            im   = Image.open(path).convert("RGBA")
            qimg = QImage(im.tobytes("raw", "RGBA"), im.width, im.height,
                          QImage.Format_RGBA8888)
            pix  = QPixmap.fromImage(qimg)
        except Exception:
            return None
    return pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
