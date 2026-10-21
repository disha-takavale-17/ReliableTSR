"""Unit tests for ephemeral storage, isolation, and auto-purge."""
import pytest
from PIL import Image
from reliabletsr.storage.sandbox import EphemeralStorage


def test_ephemeral_storage_lifecycle():
    storage = EphemeralStorage(max_file_size_mb=1, max_docs_per_user=2)
    session_a = "session_user_a"
    session_b = "session_user_b"

    dummy_image = Image.new("RGB", (50, 50), color="white")
    payload = b"dummy file bytes"

    # Store doc for session A
    storage.store_document(session_a, "doc_1", "test.pdf", payload, [dummy_image])
    assert storage.get_document(session_a, "doc_1") is not None

    # Verify session B cannot see session A's document (Isolation)
    assert storage.get_document(session_b, "doc_1") is None
    assert len(storage.list_documents(session_b)) == 0

    # Purge single document
    assert storage.purge_document(session_a, "doc_1") is True
    assert storage.get_document(session_a, "doc_1") is None

    # Test auto-purge session
    storage.store_document(session_a, "doc_2", "test2.pdf", payload, [dummy_image])
    storage.purge_session(session_a)
    assert len(storage.list_documents(session_a)) == 0


def test_storage_quota_enforcement():
    storage = EphemeralStorage(max_file_size_mb=1, max_docs_per_user=2)
    session_id = "test_quota"
    dummy_image = Image.new("RGB", (10, 10))

    # Test file size limit exceeded (> 1MB)
    oversized_payload = b"0" * (1024 * 1024 + 10)
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        storage.store_document(session_id, "big", "big.pdf", oversized_payload, [dummy_image])
