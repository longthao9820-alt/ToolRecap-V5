"""Tests for Windows DPAPI secret storage."""

import os
from pathlib import Path
import sys
import unittest.mock as mock

import pytest

from toolrecap_v4.errors import DPAPIError, SecretExposureError
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.secrets import (
    DPAPISecretStore,
    dpapi_decrypt,
    dpapi_encrypt,
)


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI tests require Windows")
def test_dpapi_encrypt_decrypt_roundtrip():
    """Verify DPAPI encrypt and decrypt roundtrip on Windows."""
    secret_bytes = b"my_super_secret_api_token_12345"
    cipher = dpapi_encrypt(secret_bytes)
    assert cipher != secret_bytes
    assert b"my_super_secret_api_token_12345" not in cipher

    decrypted = dpapi_decrypt(cipher)
    assert decrypted == secret_bytes


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI tests require Windows")
def test_dpapi_secret_store_crud(tmp_path: Path):
    """Verify DPAPISecretStore CRUD operations."""
    store = DPAPISecretStore(storage_root=tmp_path)

    # Initial state
    assert store.get_secret("gateway_key") is None
    assert not store.has_secret("gateway_key")
    assert store.list_keys() == []

    # Set secrets
    store.set_secret("gateway_key", "sec-gw-9999")
    store.set_secret("remote_voice_key", "sec-voice-8888")

    assert store.has_secret("gateway_key")
    assert store.has_secret("remote_voice_key")
    assert store.get_secret("gateway_key") == "sec-gw-9999"
    assert store.get_secret("remote_voice_key") == "sec-voice-8888"
    assert store.list_keys() == ["gateway_key", "remote_voice_key"]

    # Verify bytes on disk are not plaintext
    raw_disk_bytes = store.secrets_file.read_bytes()
    assert b"sec-gw-9999" not in raw_disk_bytes
    assert b"sec-voice-8888" not in raw_disk_bytes

    # Delete secret
    assert store.delete_secret("gateway_key") is True
    assert not store.has_secret("gateway_key")
    assert store.get_secret("gateway_key") is None
    assert store.delete_secret("gateway_key") is False

    # Clear
    store.clear()
    assert not store.secrets_file.exists()
    assert store.list_keys() == []


def test_dpapi_non_windows_rejection():
    """Verify DPAPI strictly rejects non-Windows platforms with no plaintext fallback."""
    with mock.patch("sys.platform", "linux"):
        with pytest.raises(DPAPIError, match="Windows DPAPI is required"):
            dpapi_encrypt(b"secret")

        with pytest.raises(DPAPIError, match="Windows DPAPI is required"):
            dpapi_decrypt(b"cipher")


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI tests require Windows")
def test_dpapi_corrupted_cipher_rejection(tmp_path: Path):
    """Verify corrupted ciphertext raises DPAPIError and does not fall back to plaintext."""
    store = DPAPISecretStore(storage_root=tmp_path)
    store.secrets_file.write_bytes(b"corrupted_garbage_bytes_not_dpapi")

    with pytest.raises(DPAPIError):
        store.get_secret("any_key")


def test_secrets_rejected_in_project_persistence(tmp_path: Path):
    """Verify that secrets cannot be saved into project state or settings JSON."""
    persistence = ProjectPersistence(storage_root=tmp_path)

    # Attempt to save settings with secret key must fail
    bad_settings = {"gateway_endpoint": "http://127.0.0.1:20128", "api_key": "leaked_secret"}
    with pytest.raises(SecretExposureError):
        persistence.save_settings(bad_settings)

    # Attempt to save project with secret must fail
    bad_project = {"project_id": "test_proj", "secret_token": "leaked_secret"}
    with pytest.raises(SecretExposureError):
        persistence.save_project(bad_project)
