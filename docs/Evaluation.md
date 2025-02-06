# Evaluation

Here we demonstrate the results of our evaluation exercise.

## Supermat Evaluation Metrics

> Results of evaluation on Supermat retriever.

{{ read_csv('assets/supermat_benchmarks_paragraph_chunks.csv') }}

## SOTA Comparison

> Aggregated Results Difference compared with Langchain [SemanticChunker Percentile](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)

{{ read_csv('assets/supermat_benchmarks_paragraph_chunks_semantic_diff.csv') }}

## Baseline Comparison

> Aggregated Results Difference compared with Langchain [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)

{{ read_csv('assets/supermat_benchmarks_paragraph_chunks_baseline_diff.csv') }}
