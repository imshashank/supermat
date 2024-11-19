from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supermat.core.models.base_chunk import DocumentChunksType
    from supermat.core.models.parsed_document import ParsedDocumentType


class BaseChunker(ABC):
    @abstractmethod
    def create_chunks(self, procssed_document: ParsedDocumentType) -> DocumentChunksType:  # noqa: U100
        ...
