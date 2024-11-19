from pathlib import Path

import pytest

from supermat.core import ParsedDocumentType, load_parsed_document
from supermat.core.chunking.simple_chunking import SimpleChunker
from supermat.core.models.base_chunk import ChunkDocument, DocumentChunksType
from supermat.core.models.parsed_document import BaseTextChunk, ChunkModelType


@pytest.fixture(scope="session")
def parsed_document(test_json: Path) -> ParsedDocumentType:
    return load_parsed_document(test_json)


@pytest.fixture(scope="session")
def document_chunks(parsed_document: ParsedDocumentType) -> DocumentChunksType:
    return SimpleChunker().create_chunks(parsed_document)


def same_doc(parsed_document: ChunkModelType, chunk_document: ChunkDocument) -> bool:
    return (
        parsed_document.text == chunk_document.text and parsed_document.structure == chunk_document.metadata.structure
    )


def test_simple_chunking(parsed_document: ParsedDocumentType, document_chunks: DocumentChunksType):
    text_docs = [doc for doc in parsed_document if isinstance(doc, BaseTextChunk)]
    assert len(text_docs) == len(document_chunks)
    assert all(same_doc(parsed_doc, chunk_doc) for parsed_doc, chunk_doc in zip(text_docs, document_chunks))
