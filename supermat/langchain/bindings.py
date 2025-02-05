from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
from langchain_core.language_models import BaseLanguageModel
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

    def _create_document_index(self) -> tuple[dict[str, int], dict[int, str]]:
        documents = {
            chunk.document
            for chunk in self.parsed_docs
            # NOTE: we assume that all chunks have document
            if chunk.document is not None
        }
        # NOTE: we want the document id to start with 1, since 0 means all in structure id.
        document_index_map = {document: doc_id for doc_id, document in enumerate(documents, 1)}
        index_document_map = dict(zip(document_index_map.values(), document_index_map.keys()))
        return document_index_map, index_document_map

    def _add_doc_id(self, document_index_map: dict[str, int]):
        """
        Mutates current `parsed_docs` to include document id in the chunk structure id.
        This is a temporary solution.
        Currently, the parsed documents do not include document as part of the strucutre id.
        We include document id in the relevant retrieved documents for now.
        TODO (@legendof-selda): Include document id as part of structure id in `ParsedDocumentType`.

        Args:
            document_index_map (dict[str, int]): 'document' name to index mapping.

        """
        for chunk in self.parsed_docs:
            assert chunk.document
            doc_index = document_index_map[chunk.document]
            chunk.structure = f"{doc_index}.{chunk.structure}"

        return self.parsed_docs

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        # TODO (@legendof-selda): integrate the chunker class here instead.
        # TODO (@legendof-selda): Build reverse lookups to get higher level sections easily from parsed_docs.
        self._document_index_map, self._index_document_map = self._create_document_index()
        self._add_doc_id(self._document_index_map)
        # NOTE: Currently paragraph chunks seemed to work best instead of sentence.
        self.vector_store.add_documents(
            [
                Document(
                    sentence.text,
                    metadata=dict(
                        document=chunk.document,
                        structure=sentence.structure,
                        # properties=chunk.properties,
                        key=",".join(sentence.key),
                        citation_id=sentence.structure,
                    ),
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
                    metadata=dict(
                        document=chunk.document,
                        structure=chunk.structure,
                        # properties=chunk.properties,
                        key=",".join(chunk.key),
                        citation_id=chunk.structure,
                    ),
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
                    document=chunk.document,
                    # properties=chunk.properties,
                    key=",".join(chunk.key),
                    citation_id=chunk.structure,
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
        if self.store_sentences:
            documents = self._get_higher_section(documents)
        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        documents = await self.vector_store_retriver._aget_relevant_documents(query, run_manager=run_manager)
        if self.store_sentences:
            documents = self._get_higher_section(documents)
        return documents


def format_docs(docs: list[Document]) -> str:
    response = ["{{" f"'text':'{doc.page_content}', 'metadata': '{json.dumps(doc.metadata)}', " "}}" for doc in docs]
    return f"[{','.join(response)}]"


def pre_format_docs(docs: list[Document]) -> list[Document]:
    return [
        Document(page_content=doc.page_content, metadata=(doc.metadata | {"citation_id": doc.metadata["citation_id"]}))
        for doc in docs
    ]


def get_default_prompt() -> ChatPromptTemplate:
    system_prompt = (
        "From the given list of documents used as context, choose the right documents for the given question. "
        "Answer only from the context provided and cite them as well using the given `citation_id`. "
        "This is how you should cite the text directly but using this cite block `<cite ref='citation_id'/>`."
        "Quote directly from context by specifying the `citation_id` inside the cite block to quote directly instead "
        "of writing it again. "
        "Here is the context in backticks: ```\n{context}\n```\n"
        "If you don't know the answer, say you don't know. "
        "Only answer from the context. "
        "Use three sentence maximum and keep the answer concise, but make sure the cite block is used when answering. "
        "Answer the following question, using the cite block `<cite />` instead of answering directly. "
        "```\n{question}\n```"
        "Remeber to use the cite block like this "
        "`<cite ref='citation_id' />` when thinking."
        "Example: If the answer is in a document with `citation_id='5.2.3.0'`, "
        "you must a cite block like this `<cite ref='5.2.3.0' />`. "
        "Think through how to use the `<cite />` block like shown above. "
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    prompt = ChatPromptTemplate.from_template(system_prompt)
    return prompt


@dataclass(eq=True, frozen=True)
class ParsedCite:
    ref: str
    cite_block: str
    start: int = 0
    end: int | None = None


def parse_cite_blocks(text: str) -> tuple[ParsedCite]:
    """Parses the `<cite ref='citation_id', start=0, end=None />` cite block in a text.
    NOTE: Could not get the LLM to return start and end via prompt templating.\
    This is a demo to show that citations are possible, and thus reduces output tokens from llm.
    With citations, we can avoid llm's returning tokens which are already available in context.

    Args:
        text (str): Text containing the cite block

    Returns:
        tuple[ParsedCite]: Parsed citations found in text.
    """
    pattern = r"(<cite ref='([^']*)'(,?\s*start=(\d+))?(,?\s*end=(\d+))?\s*/>)"
    matches = re.findall(pattern, text)
    parsed_blocks = set()
    for match in matches:
        block = {"cite_block": match[0], "ref": match[1]}
        if match[3]:
            block["start"] = int(match[3])
        if match[5]:
            block["end"] = int(match[5])
        parsed_blocks.add(ParsedCite(**block))

    return tuple(parsed_blocks)


class ChainOutput(TypedDict):
    llm_output: str
    context: list[Document]


def post_process(chain_output: ChainOutput, substitute: bool = False) -> str:
    """LLM chain link to replace references with content using regex matching.

    Args:
        chain_output (ChainOutput): Output of previous Lanchain link.
        substitute (bool, optional): Substitute the reference matched directly. Defaults to False.
            If, substitute is False, in a new paragraph, the referenced chunk is dumped directly.

    Returns:
        str: Returns model output with reference ids parsed to actual content.
    """
    output = chain_output["llm_output"]
    doc_mapping = {doc.metadata["citation_id"]: doc for doc in chain_output["context"]}

    references_used = parse_cite_blocks(output)
    total_references = len(references_used)

    def get_reference_quote(parsed_cite: ParsedCite) -> str | None:
        if parsed_cite.ref not in doc_mapping:
            return None
        reference = doc_mapping[parsed_cite.ref]
        return reference.page_content[parsed_cite.start : parsed_cite.end]

    if substitute:
        processed_output = output
        for parsed_cite in references_used:
            reference = (
                ref if (ref := get_reference_quote(parsed_cite)) else f"<invalid_ref# {parsed_cite.cite_block}/>"
            )
            processed_output = processed_output.replace(parsed_cite.cite_block, reference, 1)
    else:

        def get_reference(parsed_cite: ParsedCite) -> str:
            return (
                ref + "\n" + pformat(doc_mapping[parsed_cite.ref].model_dump())
                if (ref := get_reference_quote(parsed_cite))
                else f"[Reference not found: {parsed_cite.cite_block}]"
            )

        references = [
            f"{i:0{total_references}}. {parsed_cite.cite_block}:\n\t{get_reference(parsed_cite)}\n"
            for i, parsed_cite in enumerate(references_used, 1)
        ]
        references_section = "\n\nReferences:\n" + ("\n".join(references))

        processed_output = output + references_section

    return processed_output


def get_default_chain(
    retriever: SupermatRetriever,
    llm_model: BaseLanguageModel,
    substitute_references: bool = False,
    return_context: bool = False,
) -> RunnableSerializable:
    """Default chain that implements citation where LLM returns the referenced id as well
    instead of directly returning the values verbatim. This saves output tokens being generated and the actual content
    is returned during post processing.

    Args:
        retriever (SupermatRetriever): SupermatRetriever that retrieves the relevant document chunks for LLM context.
        llm_model (BaseChatModel | BaseLLM): The LLM model used for inference
        substitute_references (bool, optional): Whether to replace the citations direction, or as a separate section.
            Defaults to False.
        return_context (bool, optional): Return retrived documents for debugging. Defaults to False.

    Returns:
        RunnableSerializable: Langchain chain to run prompt query.
    """
    prompt = get_default_prompt()
    chain = RunnableParallel({"context": retriever | pre_format_docs, "question": RunnablePassthrough()}) | {
        "llm_output": prompt.partial(context=itemgetter("context") | RunnableLambda(format_docs))
        | llm_model
        | StrOutputParser(),
        "context": itemgetter("context"),
    }
    _post_process = RunnableLambda(partial(post_process, substitute=substitute_references))
    if return_context:
        chain |= {
            "llm_output": itemgetter("llm_output"),
            "answer": _post_process,
            "context": itemgetter("context"),
            "formatted_context": (itemgetter("context") | RunnableLambda(format_docs)),
        }
    else:
        chain |= _post_process

    return chain
