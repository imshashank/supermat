# Retrieval System

The Supermat Retriever integrates seamlessly with existing retrieval frameworks while leveraging our structured document approach. Here's how it works and what makes it unique.

## Supermat Retriever

Our retriever is designed as a flexible wrapper that enhances existing vector store capabilities with Supermat's structural awareness. It currently integrates with [Langchain's Vector Stores](https://python.langchain.com/docs/concepts/vectorstores/), serving as a drop-in replacement that preserves all standard functionality while adding structural context.

### Key Features

- **Seamless Integration**: Works as a direct replacement for any Langchain vector store
- **Structure-Aware Retrieval**: Maintains document hierarchies during the retrieval process
- **Framework Flexibility**: Built to support multiple retrieval frameworks
- **Preserved Context**: Utilizes Structure IDs to maintain document relationships

### Implementation

To use the Supermat Retriever:

1. Process all the documents using the `FileProcessor`
2. Choose your preferred vector store
3. Initialize `SupermatRetriever` with your selected store and the processed documents
4. Process and retrieve documents while maintaining structural integrity

Example:

#### Step 1: Process all your files

```python
from itertools import chain
from typing import TYPE_CHECKING, cast

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from supermat.core.parser import FileProcessor

parsed_files = Parallel(n_jobs=-1, backend="threading")(delayed(FileProcessor.parse_file)(path) for path in pdf_files)

documents = list(chain.from_iterable(parsed_docs for parsed_docs in parsed_files))
```

#### Step 2: Take the processed documents and build the retriever

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from supermat.langchain.bindings import SupermatRetriever

retriever = SupermatRetriever(
    parsed_docs=documents,
    vector_store=Chroma(
        embedding_function=HuggingFaceEmbeddings(
            model_name="thenlper/gte-base",
        ),
        persist_directory="./chromadb",
        collection_name="NAME",
    ),
)
```

#### Step 3. Now the retriever can be used in a Langchain retrieval chain.

You can use the default chain that we provide that provides citation. This prompt was built on Deepseek 8b model.

```python
from langchain_ollama.llms import OllamaLLM

from supermat.langchain.bindings import get_default_chain

llm_model = OllamaLLM(model="deepseek-r1:8b", temperature=0.0)
chain = get_default_chain(retriever, llm_model, substitute_references=False, return_context=False)
```
