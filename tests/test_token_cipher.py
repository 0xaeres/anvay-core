"""TokenCipher — Fernet encryption for connector tokens at rest."""

from __future__ import annotations

import pytest

from anvay.auth.token_cipher import TokenCipher, TokenCipherError


def test_encrypt_decrypt_round_trip() -> None:
    cipher = TokenCipher(TokenCipher.generate_key())
    secret = "PLAINTEXT_SECRET_TOKEN_value_123"
    encrypted = cipher.encrypt(secret)
    assert encrypted != secret
    assert cipher.decrypt(encrypted) == secret


def test_invalid_key_raises() -> None:
    with pytest.raises(TokenCipherError):
        TokenCipher("not-a-valid-fernet-key")


def test_wrong_key_cannot_decrypt() -> None:
    a = TokenCipher(TokenCipher.generate_key())
    b = TokenCipher(TokenCipher.generate_key())
    encrypted = a.encrypt("secret")
    with pytest.raises(TokenCipherError):
        b.decrypt(encrypted)


def test_generate_key_is_usable() -> None:
    key = TokenCipher.generate_key()
    assert isinstance(key, str)
    # round-trips through a fresh cipher built from the generated key
    assert TokenCipher(key).decrypt(TokenCipher(key).encrypt("x")) == "x"
