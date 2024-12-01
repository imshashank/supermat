from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from functools import wraps
from pathlib import Path

from langchain.schema.vectorstore import VectorStore
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from supermat.core.models.parsed_document import BaseTextChunk, ParsedDocumentType
from supermat.core.parser import FileProcessor


class SupermatDocLoader(BaseLoader):
    """
    Load document via Supermat parser.
    This loader is meant to be used with the Supermat Langchain componenets only and not as standalone.
    The document returned by this class is a json stringified version that is meant to used later internally.
    """

    def __init__(self, file_path: Path, include_non_text_chunks: bool):
        self._parsed_document = FileProcessor.parse_file(file_path)
        self.include_non_text_chunks = include_non_text_chunks

    @property
    def parsed_document(self) -> ParsedDocumentType:
        return self._parsed_document

    def lazy_load(self) -> Iterator[Document]:
        yield from (
            Document(page_content=chunk.model_dump_json())
            for chunk in self.parsed_document
            if self.include_non_text_chunks or isinstance(chunk, BaseTextChunk)
        )


class DelegateToBaseStoreMeta(ABCMeta):
    def __new__(cls, name, bases, namespace, **kwargs):
        # Collect all abstract methods
        abstract_methods = {attr for base in bases for attr in getattr(base, "__abstractmethods__", set())}

        # Add methods from base_store if not explicitly defined
        for method in abstract_methods:
            if method not in namespace:

                def delegate_method(self, *args, _method=method, **kwargs):
                    base_method = getattr(self.base_store, _method)
                    return base_method(*args, **kwargs)

                namespace[method] = delegate_method

        return super().__new__(cls, name, bases, namespace, **kwargs)


class SupermatVectorStore(VectorStore, metaclass=DelegateToBaseStoreMeta):
    def __init__(self, base_store: VectorStore):
        self.base_store = base_store
