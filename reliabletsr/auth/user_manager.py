"""Secure user management with PBKDF2 password hashing and role-based access."""
import hashlib
import os
import secrets
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class User:
    username: str
    role: str
    email: str
    salt: str
    password_hash: str


class UserManager:
    """Manages user authentication and verification using secure PBKDF2 hashing."""

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._init_default_users()

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash a password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
        if not salt:
            salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        ).hex()
        return pwd_hash, salt

    def _init_default_users(self):
        """Preconfigure standard enterprise accounts (Admin, Analyst, Demo)."""
        defaults = [
            ("admin", "Admin@Secure2026!", "admin", "admin@enterprise.internal"),
            ("analyst", "Analyst@DocAI!", "analyst", "analyst@enterprise.internal"),
            ("demo", "DemoPass123!", "viewer", "demo@enterprise.internal"),
        ]
        for username, password, role, email in defaults:
            pwd_hash, salt = self.hash_password(password)
            self._users[username] = User(
                username=username,
                role=role,
                email=email,
                salt=salt,
                password_hash=pwd_hash,
            )

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user against stored PBKDF2 hash using constant-time comparison."""
        user = self._users.get(username.strip().lower())
        if not user:
            return None
        candidate_hash, _ = self.hash_password(password, salt=user.salt)
        if secrets.compare_digest(user.password_hash, candidate_hash):
            return user
        return None

    def register_user(self, username: str, password: str, role: str = "viewer", email: str = "") -> bool:
        """Register a new user in memory with validated password strength."""
        clean_user = username.strip().lower()
        if clean_user in self._users or len(password) < 6:
            return False
        pwd_hash, salt = self.hash_password(password)
        self._users[clean_user] = User(
            username=clean_user,
            role=role,
            email=email or f"{clean_user}@enterprise.internal",
            salt=salt,
            password_hash=pwd_hash,
        )
        return True
