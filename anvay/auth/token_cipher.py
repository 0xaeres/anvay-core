"""Symmetric encryption for connector tokens at rest.

Tokens are encrypted with Fernet (AES-128-CBC + HMAC). The key comes from the
`ANVAY_TOKEN_KEY` env var — generate one with `TokenCipher.generate_key()`.
When no key is configured, registry writes that include secret-bearing
connector config fail clearly rather than storing secrets in plaintext.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken


class TokenCipherError(RuntimeError):
    pass


class TokenCipher:
    def __init__(self, key: str | bytes):
        raw = key.encode() if isinstance(key, str) else key
        try:
            self._fernet = Fernet(raw)
        except (ValueError, TypeError) as e:
            raise TokenCipherError(
                "invalid ANVAY_TOKEN_KEY — must be a urlsafe-base64 32-byte Fernet "
                "key; generate one with TokenCipher.generate_key()"
            ) from e

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as e:
            raise TokenCipherError(
                "could not decrypt a stored token — the ANVAY_TOKEN_KEY may have "
                "changed since it was written"
            ) from e

    @staticmethod
    def generate_key() -> str:
        """A fresh Fernet key — put this in ANVAY_TOKEN_KEY."""
        return Fernet.generate_key().decode()
