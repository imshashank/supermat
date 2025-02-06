import base64
from pathlib import Path
from typing import Any

import orjson
import pymupdf

from supermat.core.parser.pymupdf_parser.pymupdf_internal_model import PyMuPDFDocument


def default(obj):
    if isinstance(obj, pymupdf.Rect):
        return obj.__dict__
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    raise TypeError


def create_page(page: pymupdf.Page) -> dict[str, Any]:
    page_data = page.get_text("dict", sort=True)["blocks"]  # pyright: ignore[reportAttributeAccessIssue]
    page_text = page.get_text()  # pyright: ignore[reportAttributeAccessIssue]
    return {"number": page.number, "rect": page.rect, "text": page_text, "blocks": page_data}


def parse_pdf(pdf_file: Path) -> PyMuPDFDocument:
    """Converts pdf file to a PyMuPDF Document model to easy parsing.
    pymupdf, doesn't provide a pydantic model in their implementation.
    We convert it into a pydantic model to make it easier to work with.

    Args:
        pdf_file (Path): The pdf file that needs to be parsed.

    Returns:
        PyMuPDFDocument: Pydantic model representation of the pdf file.
    """
    doc = pymupdf.open(pdf_file)
    doc_data = {"filename": pdf_file.name, "total_pages": len(doc), "pages": [create_page(page) for page in doc]}
    return PyMuPDFDocument.model_validate_json(orjson.dumps(doc_data, default=default))
