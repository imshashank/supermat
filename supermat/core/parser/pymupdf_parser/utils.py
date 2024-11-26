import base64
from pathlib import Path

import orjson
import pymupdf

from supermat.core.parser.pymupdf_parser.pymupdf_internal_model import PyMuPDFDocument


def default(obj):
    if isinstance(obj, pymupdf.Rect):
        return obj.__dict__
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    raise TypeError


def create_page(page: pymupdf.Page) -> dict:
    page_data = page.get_text("dict", sort=True)["blocks"]
    return {"number": page.number, "rect": page.rect, "text": page.get_text(), "blocks": page_data}


def parse_pdf(pdf_file: Path) -> PyMuPDFDocument:
    doc = pymupdf.open(pdf_file)
    doc_data = {"filename": pdf_file.name, "total_pages": len(doc), "pages": [create_page(page) for page in doc]}
    return PyMuPDFDocument.model_validate_json(orjson.dumps(doc_data, default=default))
