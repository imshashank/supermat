from __future__ import annotations

import json
import re
from functools import cached_property, partial
from operator import itemgetter
from pprint import pformat
from typing import Any, TypedDict

from langchain.schema.vectorstore import VectorStore, VectorStoreRetriever
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSerializable,
)
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
    Args:
        parsed_docs (ParsedDocumentType): The supermat parsed documents.
        vector_store (VectorStore): The vector store used to store the document chunks.
        vector_store_retriver_kwargs (dict[str, Any], optional): `VectorStore` kwargs used during initialization.
            Defaults to `{}`.
        max_chunk_length (int, optional): Max character length. NOTE: This needs to be based on tokens instead.
            Defaults to 8000.
        store_sentences (bool, optional): Store sentence level chunks in vector store
            which will then be converted to paragraphs before sending to LLM. Defaults to False.
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


REFERENCE_PATTERN = "<ref={id}/>"


def format_docs(docs: list[Document]) -> str:
    response = [f"{{'text':{doc.page_content}, 'metadata': {json.dumps(doc.metadata)}}}" for doc in docs]
    return str(response)


def get_default_prompt() -> ChatPromptTemplate:
    ref = REFERENCE_PATTERN.format(id="id")
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Directly quote from the context given by returning the document reference "
        f"(in this format `{ref}`) found in metadata when required. "
        f"No need to quote it verbatim, just provide the reference like this format `{ref}`. "
        f"You must use the exact reference format {ref} when referring to documents "
        f"and must work with this regex pattern {REFERENCE_PATTERN.format(id=r'(.*?)')}.\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    return prompt


def format_ref(doc_id: str) -> str:
    return REFERENCE_PATTERN.format(id=doc_id)


class ChainOutput(TypedDict):
    llm_output: str
    context: list[Document]


def post_process(chain_output: ChainOutput, substitute: bool = True) -> str:
    """LLM chain link to replace references with content using regex matching.

    Args:
        chain_output (ChainOutput): Output of previous Lanchain link.
        substitute (bool, optional): Substitute the reference matched directly. Defaults to True.
            If, substitute is False, in a new paragraph, the referenced chunk is dumped directly.

    Returns:
        str: Returns model output with reference ids parsed to actual content.
    """
    output = chain_output["llm_output"]
    doc_mapping = {doc.metadata["id"]: doc for doc in chain_output["context"]}

    pattern = REFERENCE_PATTERN.format(id=r"(.*?)")  # r'<ref=(.*?)/>'

    if substitute:

        def replace_match(match):
            doc_id = match.group(1)
            doc = doc_mapping.get(doc_id, None)
            return doc.page_content if doc else f"<invalid_ref# {doc_id}/>"

        # Replace all references in a single pass
        processed_output = re.sub(pattern, replace_match, output)
    else:
        # there can be duplicate references, so take unique set.
        references_used = tuple(set(re.findall(pattern, output)))
        total_references = len(references_used)

        def get_reference(reference_id: str) -> str:
            return (
                pformat(reference.model_dump())
                if (reference := doc_mapping.get(reference_id, None))
                else f"[Reference not found: {reference_id}]"
            )

        references = [
            f"{i:0{total_references}}. {format_ref(reference_id)}:\n\t{get_reference(reference_id)}\n"
            for i, reference_id in enumerate(references_used, 1)
        ]
        references_section = "\n\nReferences:\n" + ("\n".join(references))

        processed_output = output + references_section

    return processed_output


def get_default_chain(
    retriever: SupermatRetriever,
    llm_model: BaseChatModel | BaseLLM,
    substitute_references: bool = True,
    return_context: bool = False,
) -> RunnableSerializable:
    """Default chain that implements citation where LLM returns the referenced id as well
    instead of directly returning the values verbatim. This saves output tokens being generated and the actual content
    is returned during post processing.

    Args:
        retriever (SupermatRetriever): SupermatRetriever that retrieves the relevant document chunks for LLM context.
        llm_model (BaseChatModel | BaseLLM): The LLM model used for inference
        substitute_references (bool, optional): Whether to replace the citations direction, or as a separate section.
            Defaults to True.
        return_context (bool, optional): Return retrived documents for debugging. Defaults to False.

    Returns:
        RunnableSerializable: Langchain chain to run prompt query.
    """
    chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | get_default_prompt()
        | llm_model
        | StrOutputParser()
    )
    prompt = get_default_prompt()
    chain = RunnableParallel({"context": retriever, "question": RunnablePassthrough()}) | {
        "llm_output": prompt.partial(context=itemgetter("context") | RunnableLambda(format_docs))
        | llm_model
        | StrOutputParser(),
        "context": itemgetter("context"),
    }
    _post_process = RunnableLambda(partial(post_process, substitute=substitute_references))
    if return_context:
        chain |= {"answer": _post_process, "context": itemgetter("context")}
    else:
        chain |= _post_process

    return chain
