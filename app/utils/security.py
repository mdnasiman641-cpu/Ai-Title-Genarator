"""API-key protection: encryption at rest, masking and log redaction.

On Windows keys are encrypted with DPAPI (``CryptProtectData``), the same
mechanism Windows Credential Manager and browsers use: the ciphertext can only
be decrypted by the same Windows user on the same machine. API keys are never
written in clear text and never hard-coded.

On other platforms (development / CI only - the product targets Windows) a
per-install random key file with owner-only permissions is used to obfuscate the
value. That fallback is clearly *weaker* than DPAPI and is documented as such.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sys
import threading
from pathlib import Path
from typing import Protocol

from app.config.constants import IS_WINDOWS

MASK_CHAR = "*"
_KEY_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),  # OpenAI-style keys
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)([\"'\s:=]+)[A-Za-z0-9_\-\.]{12,}"),
)
_registered: set[str] = set()
_lock = threading.Lock()


# ----------------------------------------------------------------- masking
def mask_key(key: str | None) -> str:
    """Return ``************ABCD`` for a stored key; never the full value."""
    if not key:
        return ""
    tail = key[-4:] if len(key) >= 8 else ""
    return MASK_CHAR * 12 + tail


def register_secret(secret: str | None) -> None:
    """Remember a secret so :func:`redact` can scrub it from any text."""
    if secret and len(secret) >= 6:
        with _lock:
            _registered.add(secret)


def redact(text: str) -> str:
    """Remove API keys (known patterns and registered secrets) from ``text``."""
    with _lock:
        secrets_now = tuple(_registered)
    for secret in secrets_now:
        text = text.replace(secret, "[REDACTED]")
    text = _KEY_PATTERNS[0].sub("[REDACTED]", text)
    text = _KEY_PATTERNS[1].sub("[REDACTED]", text)
    return _KEY_PATTERNS[2].sub(r"\1\2[REDACTED]", text)


# ------------------------------------------------------------ crypto backends
class ProtectionBackend(Protocol):
    """Encrypts/decrypts small byte strings for storage."""

    name: str

    def protect(self, data: bytes) -> bytes: ...

    def unprotect(self, data: bytes) -> bytes: ...


class DpapiBackend:
    """Windows Data Protection API (per-user encryption)."""

    name = "dpapi"

    def __init__(self) -> None:
        if not IS_WINDOWS:  # pragma: no cover - guarded by caller
            raise OSError("DPAPI is only available on Windows")

    @staticmethod
    def _blob_type():  # type: ignore[no-untyped-def]
        import ctypes
        from ctypes import wintypes

        class DataBlob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        return DataBlob

    def _call(self, data: bytes, encrypt: bool) -> bytes:  # pragma: no cover - Windows only
        import ctypes

        blob_cls = self._blob_type()
        buffer = ctypes.create_string_buffer(data, len(data))
        blob_in = blob_cls(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
        blob_out = blob_cls()
        crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        func = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData
        ok = func(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out))
        if not ok:
            raise OSError("Windows could not process the stored secret.")
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            kernel32.LocalFree(blob_out.pbData)

    def protect(self, data: bytes) -> bytes:  # pragma: no cover - Windows only
        return self._call(data, True)

    def unprotect(self, data: bytes) -> bytes:  # pragma: no cover - Windows only
        return self._call(data, False)


class KeyFileBackend:
    """Development fallback: HMAC-SHA256 keystream keyed by an owner-only file."""

    name = "keyfile"

    def __init__(self, key_path: Path) -> None:
        self._path = key_path

    def _key(self) -> bytes:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(secrets.token_bytes(32))
        return self._path.read_bytes()

    def _stream(self, nonce: bytes, length: int) -> bytes:
        key = self._key()
        out = bytearray()
        counter = 0
        while len(out) < length:
            out += hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
            counter += 1
        return bytes(out[:length])

    def protect(self, data: bytes) -> bytes:
        nonce = secrets.token_bytes(16)
        stream = self._stream(nonce, len(data))
        return nonce + bytes(a ^ b for a, b in zip(data, stream, strict=True))

    def unprotect(self, data: bytes) -> bytes:
        nonce, body = data[:16], data[16:]
        stream = self._stream(nonce, len(body))
        return bytes(a ^ b for a, b in zip(body, stream, strict=True))


def default_backend(data_dir: Path) -> ProtectionBackend:
    """DPAPI on Windows, key-file obfuscation elsewhere."""
    if sys.platform == "win32":
        return DpapiBackend()
    return KeyFileBackend(data_dir / "secret.key")


# ---------------------------------------------------------------- key store
class KeyValueStorage(Protocol):
    """Minimal persistence interface (implemented by the SQLite settings table)."""

    def get_raw(self, key: str) -> str | None: ...

    def set_raw(self, key: str, value: str) -> None: ...

    def delete_raw(self, key: str) -> None: ...


class SecretStore:
    """Stores named secrets (API keys) encrypted via a :class:`ProtectionBackend`."""

    PREFIX = "secret."

    def __init__(self, storage: KeyValueStorage, backend: ProtectionBackend) -> None:
        self._storage = storage
        self._backend = backend

    def set(self, name: str, value: str) -> None:
        """Encrypt and persist ``value`` under ``name``."""
        value = value.strip()
        if not value:
            self.delete(name)
            return
        token = base64.b64encode(self._backend.protect(value.encode("utf-8"))).decode("ascii")
        self._storage.set_raw(self.PREFIX + name, token)
        register_secret(value)

    def get(self, name: str) -> str | None:
        """Return the decrypted secret, or ``None`` if absent/unreadable."""
        token = self._storage.get_raw(self.PREFIX + name)
        if not token:
            return None
        try:
            value = self._backend.unprotect(base64.b64decode(token)).decode("utf-8")
        except (OSError, ValueError):
            return None
        register_secret(value)
        return value

    def has(self, name: str) -> bool:
        return self._storage.get_raw(self.PREFIX + name) is not None

    def delete(self, name: str) -> None:
        self._storage.delete_raw(self.PREFIX + name)

    def masked(self, name: str) -> str:
        """Masked display form (``************ABCD``) or an empty string."""
        return mask_key(self.get(name))
