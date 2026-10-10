"""Theme-aware SVG icons.

The bundled icons use ``currentColor``; this module substitutes the requested
colour and renders them to a crisp, high-DPI ``QIcon``.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from app.utils.helpers import resource_path

_SCALE = 2  # render at 2x so icons stay sharp on high-DPI screens


@lru_cache(maxsize=256)
def _render(name: str, color: str, size: int) -> QPixmap:
    path = resource_path("icons", f"{name}.svg")
    try:
        svg = path.read_text(encoding="utf-8").replace("currentColor", color)
    except OSError:
        return QPixmap()
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    px = size * _SCALE
    image = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, px, px))
    painter.end()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(_SCALE)
    return pixmap


def get_icon(name: str, color: str = "#98a0b3", size: int = 18) -> QIcon:
    """Return the icon ``name`` painted in ``color`` (any CSS colour string)."""
    return QIcon(_render(name, color, size))


def clear_icon_cache() -> None:
    """Drop cached pixmaps (call after a theme change)."""
    _render.cache_clear()
