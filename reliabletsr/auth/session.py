"""Session manager for tracking authenticated users and enforcing multi-tenant isolation."""
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from .user_manager import User


@dataclass
class UserSession:
    session_id: str
    user: User
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def is_expired(self, max_idle_seconds: int = 3600) -> bool:
        """Check if session has expired due to inactivity."""
        return (time.time() - self.last_active) > max_idle_seconds

    def touch(self):
        """Update last active timestamp."""
        self.last_active = time.time()


class SessionManager:
    """Manages active user sessions with timeout enforcement and clean invalidation."""

    def __init__(self, max_idle_seconds: int = 3600):
        self.max_idle_seconds = max_idle_seconds
        self._sessions: Dict[str, UserSession] = {}

    def create_session(self, user: User) -> UserSession:
        """Generate a cryptographically secure session for an authenticated user."""
        session_id = secrets.token_urlsafe(32)
        session = UserSession(session_id=session_id, user=user)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Retrieve and validate session, expiring inactive ones."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        if session.is_expired(self.max_idle_seconds):
            self.terminate_session(session_id)
            return None
        session.touch()
        return session

    def terminate_session(self, session_id: str) -> bool:
        """Terminate and revoke a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
