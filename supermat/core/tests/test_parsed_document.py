from pathlib import Path

from supermat.core import load_parsed_document
from supermat.core.models.parsed_document import FootnoteChunk, ImageChunk, TextChunk


def test_load_parsed_doc(test_json: Path):
    doc = load_parsed_document(test_json)
    first_section = doc[0]
    assert isinstance(first_section, TextChunk)
    assert first_section.type_ == "Text"
    assert first_section.structure == "0.1.0"
    assert len(first_section.key)
    assert first_section.properties.Page == 0

    assert isinstance(doc[44], ImageChunk) and doc[44].structure == "7.2.0"

    assert isinstance(doc[28], FootnoteChunk) and doc[28].structure == "4.2.0"
