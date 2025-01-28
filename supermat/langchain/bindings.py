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
    NOTE: Currently this only works on Text chunks.


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
    store_sentences: bool = False

    @cached_property
    def vector_store_retriver(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever(**self.vector_store_retriver_kwargs)

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        # TODO (@legendof-selda): integrate the chunker class here instead.
        # TODO (@legendof-selda): Build reverse lookups to get higher level sections easily from parsed_docs.
        # NOTE: Currently paragraph chunks seemed to work best instead of sentence.
        self.vector_store.add_documents(
            [
                Document(
                    sentence.text,
                    metadata={"structure": sentence.structure, "id": f"{chunk.document}-{sentence.structure}"},
                )
                for chunk in self.parsed_docs
                if isinstance(chunk, BaseTextChunk)
                for sentence in (chunk.sentences if chunk.sentences else [chunk])
                if isinstance(sentence, BaseTextChunk)
            ]
            if self.store_sentences
            else [
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
        """Utility to convert lower level structure (eg. sentences) to a higher level structure (eg. paragraphs).
        We return only unique documents back.
        Eg. If there are 3 sentences of the same paragraph, we only want a single paragraph document back.

        Args:
            documents (list[Document]): Relevant documents retrieved from the vector store.

        Returns:
            list[Document]: Relevant documents from the vector store, but converted to a higher level structure.
        """
        # TODO (@legendof-selda): Refactor to make use of inverse lookups for faster higher strucutre retrieval.
        return [
            Document(
                # TODO (@legendof-selda): this max chunk clipping is only a temp solution
                # ideally the intelligent chunker class will take care of this based on token length.
                chunk.text[: self.max_chunk_length],
                metadata=dict(
                    structure=chunk.structure,
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
