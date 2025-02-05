"""
This is where the core supermat parsing logic exists.
Core deals with Supermat's parser pydantic models to define structure to the parsed documents,
chunking strategies, and parser logic to convert documents into the `ParsedDocument` model.
"""

from supermat.core.models.parsed_document import (
    ParsedDocument,
    ParsedDocumentType,
    export_parsed_document,
    load_parsed_document,
)

__all__ = [
    "export_parsed_document",
    "load_parsed_document",
    "ParsedDocument",
    "ParsedDocumentType",
]
