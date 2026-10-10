"""Right-hand details pane: thumbnail and metadata of the highlighted file."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFormLayout, QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from app.files.metadata import VideoMetadata
from app.files.scanner import FileEntry
from app.i18n import tr
from app.ui.widgets import IconBinder
from app.utils.helpers import format_duration, format_size, format_timestamp

THUMB_SIZE = QSize(232, 130)


class DetailsPanel(QFrame):
    """Shows what is known about one file. Thumbnails/metadata arrive lazily via setters."""

    def __init__(self, icons: IconBinder) -> None:
        super().__init__()
        self.setObjectName("DetailsPane")
        self.setFixedWidth(256)
        self._icons = icons
        self._entry: FileEntry | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()  # content scrolls instead of overlapping when the card is short
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        content = QWidget()
        content.setObjectName("DetailsContent")
        content.setStyleSheet("QWidget#DetailsContent { background: transparent; }")
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.thumb = QLabel()
        self.thumb.setObjectName("Thumb")
        self.thumb.setFixedSize(THUMB_SIZE)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.thumb, 0, Qt.AlignmentFlag.AlignHCenter)

        self.name = QLabel()
        self.name.setWordWrap(True)
        self.name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.name.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.name)

        form = QFormLayout()
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._labels: dict[str, QLabel] = {}
        self._values: dict[str, QLabel] = {}
        for key in ("duration", "resolution", "fps", "video", "audio", "size", "modified"):
            label, value = QLabel(), QLabel()
            label.setObjectName("Muted")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(label, value)
            self._labels[key], self._values[key] = label, value
        layout.addLayout(form)

        self.hint = QLabel()
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        layout.addStretch(1)
        self.has_probe = False
        self.has_ffmpeg = False
        self.retranslate()
        self.show_entry(None, None)

    # ------------------------------------------------------------------ public
    def set_capabilities(self, *, probe: bool, ffmpeg: bool) -> None:
        """Tell the panel which optional tools exist (controls the hint text)."""
        self.has_probe, self.has_ffmpeg = probe, ffmpeg
        self._refresh_hint()

    def show_entry(self, entry: FileEntry | None, meta: VideoMetadata | None) -> None:
        self._entry = entry
        self.set_thumbnail(None)
        if entry is None:
            self.name.setText(tr("details.none"))
            for value in self._values.values():
                value.setText("")
            return
        self.name.setText(entry.name)
        known_duration = entry.duration if entry.duration and entry.duration > 0 else None
        self._values["duration"].setText(format_duration(meta.duration if meta and meta.duration else known_duration) or "—")
        self._values["resolution"].setText((meta.resolution if meta else "") or "—")
        self._values["fps"].setText(f"{meta.fps:g}" if meta and meta.fps else "—")
        self._values["video"].setText((meta.video_codec if meta else "") or "—")
        self._values["audio"].setText((meta.audio_codec if meta else "") or "—")
        self._values["size"].setText(format_size(entry.size))
        self._values["modified"].setText(format_timestamp(entry.mtime))

    def set_thumbnail(self, image_path: str | None) -> None:
        """Show the thumbnail at ``image_path`` or the placeholder icon."""
        if image_path:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                self.thumb.setPixmap(
                    pixmap.scaled(THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
                return
        self.thumb.setPixmap(QPixmap())
        self.thumb.setText("🎬" if self._entry is not None else "")
        self.thumb.setStyleSheet("font-size: 40px;")

    @property
    def current_rel_path(self) -> str | None:
        return self._entry.rel_path if self._entry else None

    # --------------------------------------------------------------- translate
    def _refresh_hint(self) -> None:
        self.hint.setText("" if (self.has_probe and self.has_ffmpeg) else tr("details.no_ffmpeg"))

    def retranslate(self) -> None:
        for key, label in self._labels.items():
            label.setText(tr(f"details.{key}"))
        self._refresh_hint()
        if self._entry is None:
            self.name.setText(tr("details.none"))
