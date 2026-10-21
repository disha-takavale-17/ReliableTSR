"""Unit tests for user authentication and session management."""
import pytest
from reliabletsr.auth.user_manager import UserManager
from reliabletsr.auth.session import SessionManager


def test_default_user_authentication():
    manager = UserManager()
    # Test valid login
    user = manager.authenticate("admin", "Admin@Secure2026!")
    assert user is not None
    assert user.username == "admin"
    assert user.role == "admin"

    # Test invalid password
    assert manager.authenticate("admin", "WrongPassword!") is None
    # Test non-existent user
    assert manager.authenticate("unknown_user", "AnyPassword") is None


def test_user_registration():
    manager = UserManager()
    success = manager.register_user("new_analyst", "SecurePass123!", role="analyst")
    assert success is True

    # Duplicate registration should fail
    assert manager.register_user("new_analyst", "SecurePass123!") is False
    # Short password should fail
    assert manager.register_user("shorty", "123") is False

    user = manager.authenticate("new_analyst", "SecurePass123!")
    assert user is not None
    assert user.role == "analyst"


def test_session_lifecycle():
    user_mgr = UserManager()
    session_mgr = SessionManager(max_idle_seconds=2)

    user = user_mgr.authenticate("admin", "Admin@Secure2026!")
    session = session_mgr.create_session(user)
    assert session.session_id is not None

    retrieved = session_mgr.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.user.username == "admin"

    # Termination
    session_mgr.terminate_session(session.session_id)
    assert session_mgr.get_session(session.session_id) is None
