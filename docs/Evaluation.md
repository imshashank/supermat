# Evaluation

## LangChain Evaluation Overview

LangChain provides evaluation modules to assess the quality of LLM outputs, particularly for RAG systems. The evaluation process typically involves comparing the generated response against reference answers or source documents.

## Key Metrics Explained

### Faithfulness Metrics

Measures how truthful or accurate the generated response is compared to the source documents. It checks if the LLM's response contains information that is actually present in the retrieved documents and doesn't include hallucinated facts.

### Accuracy

A metric that measures whether the generated response is completely correct according to the reference answer. In RAG contexts, this often means checking if all key information from the source documents is preserved.

### Cosine Similarity

Measures the semantic similarity between the generated response and reference text by converting them into vector representations and calculating their cosine distance. Values range from -1 to 1, where 1 indicates perfect similarity.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Metrics

#### ROUGE-L (RougeLsum)

- **Overall**: Measures the longest common subsequence between generated and reference texts
- **Precision**: Percentage of words in the generated text that appear in the reference
- **Recall**: Percentage of words in the reference that appear in the generated text

#### ROUGE-1

- **Overall**: Measures overlap of unigrams (single words)
- **Precision**: Ratio of matching unigrams to total unigrams in generated text
- **Recall**: Ratio of matching unigrams to total unigrams in reference text

#### ROUGE-2

- **Overall**: Measures overlap of bigrams (pairs of consecutive words)
- **Precision**: Ratio of matching bigrams to total bigrams in generated text
- **Recall**: Ratio of matching bigrams to total bigrams in reference text

These metrics together provide a comprehensive view of RAG system performance, evaluating both factual accuracy and linguistic similarity to reference materials.

Here we demonstrate the results of our evaluation exercise.

## Supermat Evaluation Metrics

> Results of evaluation on Supermat retriever.

{{ read_csv('assets/supermat_benchmarks_paragraph_chunks.csv') }}


## SOTA Comparison

> Aggregated Results Difference compared with Langchain [SemanticChunker Percentile](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html)

{{ read_csv('assets/supermat_benchmarks_paragraph_chunks_semantic_diff.csv') }}

### Faithfulness and Accuracy

- Our method demonstrates higher faithfulness (+12.5% mean improvement)
- Better accuracy scores (+15.6% mean improvement)
- Particularly strong in the lower quartile with +55.6% improvement in faithfulness

### Semantic Similarity

- Slightly lower cosine similarity (-0.76% mean difference)
- Small range of variation (from +0.46% to -1.17%)
- Median difference of -0.28%

### ROUGE Scores

- Significant improvements in ROUGE-1 metrics:
  - F1 score: +20.7% better
  - Precision: +12.6% improvement
  - Recall: +33.3% higher
- Lower performance in ROUGE-2 scores:
  - Marginal improvement in F1 (+0.77%)
  - Decreased precision (-4.5%)
  - Lower recall (-7.8%)
- Notable improvements in ROUGE-L metrics:
  - F1 score: +22.8% better
  - Precision: +15.1% higher
  - Recall: +34.3% improvement

### Performance

- Slightly faster execution times (-0.062s mean difference)
- Variable performance improvements:
  - Best case: -0.42s improvement
  - Some cases slightly slower (+0.03s in 25th percentile)
  - Median case shows minimal difference (-0.004s)

Overall, our methodology shows substantial improvements over semantic chunking in faithfulness, accuracy, and most ROUGE metrics, particularly in ROUGE-1 and ROUGE-L scores. While there's a slight decrease in semantic similarity and ROUGE-2 metrics, the performance times remain comparable with slight improvements in most cases.

## Baseline Comparison

> Aggregated Results Difference compared with Langchain [RecursiveCharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)

{{ read_csv('assets/supermat_benchmarks_paragraph_chunks_baseline_diff.csv') }}

Here's a brief summary comparing our method with LangChain's Recursive chunking:


### Faithfulness and Accuracy

- Our method demonstrates better faithfulness (+11.8% mean improvement)
- Improved accuracy scores (+13.5% mean improvement)
- Notable improvements in lower quartiles, showing better minimum performance

### Semantic Similarity

- Very slight decrease in cosine similarity (-0.55% mean difference)
- Minimal variation in differences (range: -1.88% to +0.62%)
- Most differences concentrated near the median (-0.22%)

### ROUGE Scores

- Notable improvements in ROUGE-1 metrics:
  - F1 score: +13.1% better
  - Precision: +4.2% improvement
  - Recall: +30.8% higher
- Better ROUGE-2 performance:
  - F1 score: +5.9% improvement
  - Slightly lower precision (-0.57%)
  - Higher recall (+8.9%)
- Significant gains in ROUGE-L metrics:
  - F1 score: +17.2% better
  - Precision: +9.4% higher
  - Recall: +31.6% improvement

### Performance

- Substantially faster execution time (-0.45s mean difference)
- Very consistent performance improvements across all quartiles
- Maximum time savings of 0.64s in best cases

Overall, our methodology shows meaningful improvements over recursive chunking across most metrics, with particularly strong gains in faithfulness, accuracy, and ROUGE-L scores. The performance improvement is notably more significant compared to the semantic chunker comparison, with consistent time savings while maintaining better quality metrics.
