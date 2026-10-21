"""In-memory document rendering for PDFs and image files without touching disk."""
import io
from typing import List, Tuple
from PIL import Image


class DocumentProcessor:
    """Converts uploaded bytes (PDF or standard image) into in-memory PIL images."""

    @staticmethod
    def is_pdf(raw_bytes: bytes) -> bool:
        """Check if raw byte payload has a PDF magic header."""
        return raw_bytes.startswith(b"%PDF")

    @classmethod
    def process_bytes(cls, raw_bytes: bytes, filename: str, dpi: int = 150) -> Tuple[List[Image.Image], str]:
        """Convert input bytes into a list of PIL Images in memory.

        Returns:
            Tuple of (list_of_page_images, mime_type)
        """
        if cls.is_pdf(raw_bytes) or filename.lower().endswith(".pdf"):
            return cls._render_pdf(raw_bytes, dpi), "application/pdf"
        else:
            return cls._load_image(raw_bytes), "image/jpeg"

    @staticmethod
    def _load_image(raw_bytes: bytes) -> List[Image.Image]:
        """Load single-image payload directly into memory."""
        with io.BytesIO(raw_bytes) as buf:
            img = Image.open(buf)
            img.load()  # Ensure image bytes are fully read into RAM
            return [img.convert("RGB")]

    @staticmethod
    def _render_pdf(raw_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
        """Render PDF pages to PIL images in-memory using PyMuPDF (fitz) or pypdf."""
        pages: List[Image.Image] = []

        # Strategy 1: Try PyMuPDF - fastest, high-fidelity
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img)
            doc.close()
            return pages
        except ImportError:
            pass

        # Strategy 2: Try pdf2image (if poppler is installed)
        try:
            from pdf2image import convert_from_bytes
            return convert_from_bytes(raw_bytes, dpi=dpi)
        except (ImportError, Exception):
            pass

        # Strategy 3: Try pypdf (extract embedded images)
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            for page in reader.pages:
                for img_obj in page.images:
                    pages.append(Image.open(io.BytesIO(img_obj.data)).convert("RGB"))
            if pages:
                return pages
        except (ImportError, Exception):
            pass

        raise RuntimeError(
            "PDF rendering requires 'pymupdf' (recommended: pip install pymupdf) or 'pdf2image'."
            " Alternatively, you can upload page images directly (PNG, JPG, TIFF)."
        )
