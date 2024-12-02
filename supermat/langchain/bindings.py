from __future__ import annotations

from functools import cached_property
from typing import Any

from langchain.schema.vectorstore import VectorStore
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
    def vector_store_retriver(self):
        return self.vector_store.as_retriever(**self.vector_store_retriver_kwargs)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        # TODO (@legendof-selda): integrate the chunker class here instead.
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
        )

    def _get_higher_section(self, documents: list[Document]) -> list[Document]:
        return [
            Document(
                # this max chunk clipping is only a temp solution
                # ideally the intelligent chunker class will take care of this.
                chunk.text[: self.max_chunk_length],
                metadata=dict(
                    structure=chunk.structure,
                    properties=chunk.properties,
                    key=chunk.key,
                    id=f"{chunk.document}-{chunk.structure}",
                ),
            )
            for chunk in self.parsed_docs
            if isinstance(chunk, BaseTextChunk)
            and any(chunk.is_subsection(doc.metadata.get("structure", "")) for doc in documents)
        ]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        documents = self.vector_store_retriver._get_relevant_documents(query, run_manager=run_manager)
        return self._get_higher_section(documents)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        documents = await self.vector_store_retriver._aget_relevant_documents(query, run_manager=run_manager)
        return self._get_higher_section(documents)
