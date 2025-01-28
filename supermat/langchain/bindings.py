from __future__ import annotations

from functools import cached_property
from typing import Any

from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from supermat.core.models.parsed_document import BaseTextChunk, ParsedDocumentType


class SupermatRetriever(BaseRetriever):
    """
    Supermat Langchain Custom Retriever.
    This uses any Langchain VectorStore and overrides the documents retrieval methods to make it work for Supermat.


    ``` python
    from supermat.langchain.bindings import SupermatRetriever
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    retriever = SupermatRetriever(
        parsed_docs=FileProcessor.process_file(pdf_file_path),
        document_name=pdf_file_path.stem,
        vector_store=Chroma(
            embedding_function=HuggingFaceEmbeddings(
                model_name="thenlper/gte-base",
            )
        ),
    )
    ```
    """

    parsed_docs: ParsedDocumentType = Field(exclude=True, strict=False, repr=False)
    vector_store: VectorStore
    vector_store_retriver_kwargs: dict[str, Any] = {}
    max_chunk_length: int = 8000

    @cached_property
    def vector_store_retriver(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever(**self.vector_store_retriver_kwargs)

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        # TODO (@legendof-selda): integrate the chunker class here instead.
        self.vector_store.add_documents(
            [
                Document(
                    chunk.text,
                    metadata={
                        "structure": chunk.structure,
                        "id": f"{chunk.document}-{chunk.structure}",
                        "key": ",".join(chunk.key),
                    },
                )
                for chunk in self.parsed_docs
                if isinstance(chunk, BaseTextChunk)
            ]
        )

    def _get_higher_section(self, documents: list[Document]) -> list[Document]:
        # we only want to return unique paragraphs. so we check if the structure id was seen before!
        _seen_chunks: set[str] = set()

        def _add_structure(structure: str) -> str:
            _seen_chunks.add(structure)
            return structure

        return [
            Document(
                # this max chunk clipping is only a temp solution
                # ideally the intelligent chunker class will take care of this.
                chunk.text[: self.max_chunk_length],
                metadata=dict(
                    structure=_add_structure(chunk.structure),
                    # properties=chunk.properties,
                    key=chunk.key,
                    id=f"{chunk.document}-{chunk.structure}",
                ),
            )
            # This is in paragraph level.
            for chunk in self.parsed_docs
            if isinstance(chunk, BaseTextChunk)
            and any(
                chunk.has_subsection(doc.metadata.get("structure", ""))
                # In sentence level.
                for doc in documents
            )
        ]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        documents = self.vector_store_retriver._get_relevant_documents(query, run_manager=run_manager)
        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        documents = await self.vector_store_retriver._aget_relevant_documents(query, run_manager=run_manager)
        return documents
