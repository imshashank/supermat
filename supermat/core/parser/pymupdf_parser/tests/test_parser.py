from pathlib import Path

from supermat.core.models.parsed_document import ImageChunk
from supermat.core.parser.pymupdf_parser.parser import PyMuPDFParser


def test_parser(test_pdf: Path):
    doc = PyMuPDFParser().parse(test_pdf)
    num_images = len([chunk for chunk in doc if isinstance(chunk, ImageChunk)])
    assert num_images == 2
