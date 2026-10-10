"""Generate the application icon (PNG + multi-size ICO) with Qt. Run from the project root:

    python tools/make_icon.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen

OUT = Path(__file__).resolve().parent.parent / "app" / "resources" / "icons"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def draw(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    bg = QPainterPath()
    bg.addRoundedRect(QRectF(s * 0.04, s * 0.04, s * 0.92, s * 0.92), s * 0.22, s * 0.22)
    grad = QLinearGradient(0, 0, s, s)
    grad.setColorAt(0, QColor("#5b9bff"))
    grad.setColorAt(1, QColor("#2f56e0"))
    p.fillPath(bg, grad)
    # film frame
    p.setPen(QPen(QColor("#ffffff"), max(1.0, s * 0.045), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(s * 0.2, s * 0.26, s * 0.6, s * 0.48), s * 0.06, s * 0.06)
    # play triangle
    tri = QPainterPath()
    tri.moveTo(QPointF(s * 0.43, s * 0.38))
    tri.lineTo(QPointF(s * 0.43, s * 0.62))
    tri.lineTo(QPointF(s * 0.63, s * 0.50))
    tri.closeSubpath()
    p.fillPath(tri, QColor("#ffffff"))
    # "rename" underline accent
    p.setPen(QPen(QColor("#ffd166"), max(1.0, s * 0.055), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(s * 0.28, s * 0.84), QPointF(s * 0.72, s * 0.84))
    p.end()
    return image


def png_bytes(image: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    return bytes(ba.data())


def main() -> None:
    _app = QGuiApplication(sys.argv[:1] + ["-platform", "offscreen"])
    OUT.mkdir(parents=True, exist_ok=True)
    pngs = {size: png_bytes(draw(size)) for size in SIZES}
    (OUT / "app.png").write_bytes(pngs[256])
    header = struct.pack("<HHH", 0, 1, len(SIZES))
    offset = 6 + 16 * len(SIZES)
    entries, blobs = b"", b""
    for size in SIZES:
        data = pngs[size]
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    (OUT / "app.ico").write_bytes(header + entries + blobs)
    print("wrote", OUT / "app.png", OUT / "app.ico")


if __name__ == "__main__":
    main()
