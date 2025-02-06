from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supermat.core.models.base_chunk import DocumentChunksType
    from supermat.core.models.parsed_document import ParsedDocumentType


class BaseChunker(ABC):
    """
    Base class for all Chunker implementations.
    """

    @abstractmethod
    def create_chunks(self, processed_document: ParsedDocumentType) -> DocumentChunksType:  # noqa: U100
        """Build chunks from the given ParsedDocument into list of ChunkDocuments.
        This is the public class that is called for any chunking strategy.

        Args:
            processed_document (ParsedDocumentType): The processed document that needs to split into chunks.

        Returns:
            DocumentChunksType: The chunks built by the given strategy.
        """
