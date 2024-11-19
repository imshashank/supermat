from pathlib import Path

import orjson
import orjson.orjson
import pytest

from supermat.core import ParsedDocument, ParsedDocumentType, load_parsed_document
from supermat.core.models.parsed_document import FootnoteChunk, ImageChunk, TextChunk


@pytest.fixture(scope="session")
def parsed_document(test_json: Path) -> ParsedDocumentType:
    return load_parsed_document(test_json)


def test_load_parsed_doc(parsed_document: ParsedDocumentType):
    doc = parsed_document
    first_section = doc[0]
    assert isinstance(first_section, TextChunk)
    assert first_section.type_ == "Text"
    assert first_section.structure == "0.1.0"
    assert len(first_section.key)
    assert first_section.properties.Page == 0

    assert isinstance(doc[44], ImageChunk) and doc[44].structure == "7.2.0"

    assert isinstance(doc[28], FootnoteChunk) and doc[28].structure == "4.2.0"


def test_verify_parsing(test_json: Path, parsed_document: ParsedDocumentType):
    with test_json.open("rb") as fp:
        raw_json = orjson.loads(
            orjson.dumps(orjson.loads(fp.read()), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )

    parsed_json = orjson.loads(
        orjson.dumps(
            orjson.loads(ParsedDocument.dump_json(parsed_document)), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        )
    )

    assert raw_json == parsed_json
