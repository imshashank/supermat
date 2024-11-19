from __future__ import annotations

from functools import cached_property
from typing import TypeAlias

from pydantic import BaseModel, Field, TypeAdapter, computed_field

from supermat.core.models.parsed_document import BaseChunk, ChunkModelType


class BaseChunkMetadata(BaseChunk):
    page_number: int
    source: str | None = None
    chunk_meta: ChunkModelType = Field(exclude=True)

    @computed_field
    @cached_property
    def serialized_chunk_meta(self) -> str:
        # NOTE: probably not the best way to handle this.
        # Need to look into better approaches
        return self.chunk_meta.model_dump_json()


class ChunkDocument(BaseModel):
    document_id: int
    text: str
    metadata: BaseChunkMetadata


DocumentChunksType: TypeAlias = list[ChunkDocument]
DocumentChunks = TypeAdapter(DocumentChunksType)
