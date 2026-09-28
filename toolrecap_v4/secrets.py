"""Windows DPAPI secret storage.

Secrets are encrypted with DPAPI (CryptProtectData) under the current Windows user context.
Plaintext fallback is strictly prohibited.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

from toolrecap_v4.errors import DPAPIError
from toolrecap_v4.persistence import get_storage_root

CRYPTPROTECT_UI_FORBIDDEN = 0x1
OPTIONAL_ENTROPY = b"ToolRecapV3_DPAPI_SecretStorage_v1"


class DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _check_windows() -> None:
    if sys.platform != "win32":
        raise DPAPIError("Windows DPAPI is required for secret storage; plaintext fallback is forbidden.")


def dpapi_encrypt(data: bytes, entropy: bytes = OPTIONAL_ENTROPY) -> bytes:
    """Encrypt bytes using Windows DPAPI CryptProtectData.
    
    Raises DPAPIError on failure or non-Windows. No plaintext fallback.
    """
    _check_windows()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"dpapi_encrypt expects bytes, got {type(data).__name__}")

    try:
        crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        in_blob = DATA_BLOB(
            len(data),
            ctypes.cast(ctypes.create_string_buffer(bytes(data)), ctypes.POINTER(ctypes.c_char)),
        )
        entropy_blob = DATA_BLOB(
            len(entropy),
            ctypes.cast(ctypes.create_string_buffer(entropy), ctypes.POINTER(ctypes.c_char)),
        )
        out_blob = DATA_BLOB()

        success = crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "ToolRecapV3_Secret",
            ctypes.byref(entropy_blob),
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not success:
            err = ctypes.WinError()
            raise DPAPIError(f"DPAPI CryptProtectData failed: {err}")

        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(out_blob.pbData)
    except DPAPIError:
        raise
    except Exception as e:
        raise DPAPIError(f"DPAPI encryption error: {e}") from e


def dpapi_decrypt(cipher: bytes, entropy: bytes = OPTIONAL_ENTROPY) -> bytes:
    """Decrypt bytes using Windows DPAPI CryptUnprotectData.
    
    Raises DPAPIError on failure or non-Windows. No plaintext fallback.
    """
    _check_windows()
    if not isinstance(cipher, (bytes, bytearray)):
        raise TypeError(f"dpapi_decrypt expects bytes, got {type(cipher).__name__}")

    try:
        crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        in_blob = DATA_BLOB(
            len(cipher),
            ctypes.cast(ctypes.create_string_buffer(bytes(cipher)), ctypes.POINTER(ctypes.c_char)),
        )
        entropy_blob = DATA_BLOB(
            len(entropy),
            ctypes.cast(ctypes.create_string_buffer(entropy), ctypes.POINTER(ctypes.c_char)),
        )
        out_blob = DATA_BLOB()

        success = crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            ctypes.byref(entropy_blob),
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        )
        if not success:
            err = ctypes.WinError()
            raise DPAPIError(f"DPAPI CryptUnprotectData failed: {err}")

        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            kernel32.LocalFree(out_blob.pbData)
    except DPAPIError:
        raise
    except Exception as e:
        raise DPAPIError(f"DPAPI decryption error: {e}") from e


class DPAPISecretStore:
    """Secure secret storage backed exclusively by Windows DPAPI.
    
    Secrets are stored in a binary file encrypted via DPAPI.
    They are never stored in settings.json or project state.
    """

    def __init__(self, storage_root: Optional[Path | str] = None) -> None:
        root = get_storage_root(storage_root)
        self.secrets_dir = root / "secrets"
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        self.secrets_file = self.secrets_dir / "credentials.dpapi"

    def _read_all(self) -> Dict[str, str]:
        if not self.secrets_file.exists():
            return {}
        try:
            cipher_bytes = self.secrets_file.read_bytes()
            if not cipher_bytes:
                return {}
            plain_bytes = dpapi_decrypt(cipher_bytes)
            data = json.loads(plain_bytes.decode("utf-8"))
            if not isinstance(data, dict):
                raise DPAPIError("Decrypted secret data is not a valid JSON dictionary.")
            return data
        except DPAPIError:
            raise
        except Exception as e:
            raise DPAPIError(f"Failed to load DPAPI secret store: {e}") from e

    def _write_all(self, secrets: Dict[str, str]) -> None:
        try:
            plain_bytes = json.dumps(secrets).encode("utf-8")
            cipher_bytes = dpapi_encrypt(plain_bytes)

            tmp_path = self.secrets_dir / f"{self.secrets_file.name}.tmp.{uuid.uuid4().hex}"
            with open(tmp_path, "wb") as f:
                f.write(cipher_bytes)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self.secrets_file)
        except DPAPIError:
            raise
        except Exception as e:
            if "tmp_path" in locals() and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise DPAPIError(f"Failed to write DPAPI secret store: {e}") from e

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret by key. Returns None if key is not found."""
        secrets = self._read_all()
        return secrets.get(key)

    def set_secret(self, key: str, value: str) -> None:
        """Store a secret encrypted via DPAPI."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Secret key must be a non-empty string.")
        if not isinstance(value, str):
            raise TypeError(f"Secret value must be a string, got {type(value).__name__}")

        secrets = self._read_all()
        secrets[key] = value
        self._write_all(secrets)

    def delete_secret(self, key: str) -> bool:
        """Delete a secret. Returns True if secret existed, False otherwise."""
        secrets = self._read_all()
        if key in secrets:
            del secrets[key]
            self._write_all(secrets)
            return True
        return False

    def has_secret(self, key: str) -> bool:
        """Check if secret exists."""
        secrets = self._read_all()
        return key in secrets

    def list_keys(self) -> List[str]:
        """List all secret keys stored."""
        secrets = self._read_all()
        return sorted(list(secrets.keys()))

    def clear(self) -> None:
        """Remove all secrets."""
        if self.secrets_file.exists():
            self.secrets_file.unlink()
