"""In-memory ephemeral document storage with user isolation and auto-purge."""
import io
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from PIL import Image


@dataclass
class StoredDocument:
    doc_id: str
    filename: str
    mime_type: str
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    page_images: List[Image.Image] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class EphemeralStorage:
    """RAM-only sandbox ensuring no documents persist to permanent storage."""

    def __init__(self, max_file_size_mb: int = 15, max_docs_per_user: int = 5):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_docs_per_user = max_docs_per_user
        # Structure: {session_id: {doc_id: StoredDocument}}
        self._user_stores: Dict[str, Dict[str, StoredDocument]] = {}

    def store_document(
        self,
        session_id: str,
        doc_id: str,
        filename: str,
        raw_bytes: bytes,
        page_images: List[Image.Image],
        mime_type: str = "application/pdf",
    ) -> StoredDocument:
        """Store document pages strictly in RAM with quota validation."""
        if len(raw_bytes) > self.max_file_size_bytes:
            raise ValueError(
                f"File size ({len(raw_bytes) / 1024 / 1024:.1f} MB) exceeds maximum allowed limit "
                f"({self.max_file_size_bytes / 1024 / 1024:.0f} MB)."
            )

        user_docs = self._user_stores.setdefault(session_id, {})
        if len(user_docs) >= self.max_docs_per_user:
            # Evict the oldest document to respect memory budget
            oldest_doc_id = min(user_docs.keys(), key=lambda k: user_docs[k].created_at)
            del user_docs[oldest_doc_id]

        doc = StoredDocument(
            doc_id=doc_id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(raw_bytes),
            page_images=page_images,
        )
        user_docs[doc_id] = doc
        return doc

    def get_document(self, session_id: str, doc_id: str) -> Optional[StoredDocument]:
        """Fetch document strictly from the caller's isolated session."""
        return self._user_stores.get(session_id, {}).get(doc_id)

    def list_documents(self, session_id: str) -> List[StoredDocument]:
        """List active documents in the caller's session."""
        return list(self._user_stores.get(session_id, {}).values())

    def purge_document(self, session_id: str, doc_id: str) -> bool:
        """Securely wipe a specific document from memory."""
        if session_id in self._user_stores and doc_id in self._user_stores[session_id]:
            doc = self._user_stores[session_id].pop(doc_id)
            doc.page_images.clear()
            return True
        return False

    def purge_session(self, session_id: str):
        """Securely purge all documents and memory for an entire session."""
        if session_id in self._user_stores:
            for doc in self._user_stores[session_id].values():
                doc.page_images.clear()
            self._user_stores[session_id].clear()
            del self._user_stores[session_id]

    def purge_all(self):
        """Global purge for system reset."""
        for session_id in list(self._user_stores.keys()):
            self.purge_session(session_id)
