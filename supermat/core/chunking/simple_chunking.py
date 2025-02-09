from __future__ import annotations

from typing import TYPE_CHECKING

from supermat.core.chunking import BaseChunker
from supermat.core.models.base_chunk import BaseChunkMetadata, ChunkDocument
from supermat.core.models.parsed_document import BaseTextChunk

if TYPE_CHECKING:
    from supermat.core.models.base_chunk import DocumentChunksType
    from supermat.core.models.parsed_document import ChunkModelType, ParsedDocumentType


class SimpleChunker(BaseChunker):
    """
    A simple chunking strategy that simply takes all TextChunks in the parsed document and converts them into chunks.
    """

    @staticmethod
    def build_chunk(doc_id: int, section: ChunkModelType) -> ChunkDocument:
        assert isinstance(section, BaseTextChunk)
        assert section.properties
        return ChunkDocument(
            document_id=doc_id,
            text=section.text,
            metadata=BaseChunkMetadata(
                document=section.document,
                type=section.type_,
                structure=section.structure,
                page_number=section.properties.page,
                source=section.properties.path,
                chunk_meta=section,
            ),
        )

    def create_chunks(self, processed_document: ParsedDocumentType) -> DocumentChunksType:
        return [
            SimpleChunker.build_chunk(doc_id, section)
            for doc_id, section in enumerate(processed_document)
            if isinstance(section, BaseTextChunk)
        ]
