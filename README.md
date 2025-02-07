# ![supermat](docs/assets/supermat-logo-black-sub.png "supermat")


## Preface

Current retrieval systems face two fundamental limitations that we've accepted as normal. 

First, as they fragment information during processing, they lose natural relationships. While vector search is powerful for finding semantically related content, similarity isn't same as actual relationships. Even the most sophisticated similarity search can't fully reconstruct explicit relationships nor make implicit connections clear . Systems end up spending massive resources trying to approximate context through similarity and further post-processing that was clear and direct before processing. The result is increasingly sophisticated systems bogged down by trying to reconstruct what was there all along. 
Secondly, for the purpose of referencing, these systems use flat IDs - UUIDs and random strings - that can't express relationships, forcing them to maintain separate layers just to understand how information connects. Citations are an after-thought today.  

Our aprroach solves both these problems with a fundamental insight. 
Information has natural structure - from documents to sections to paragraphs to sentences. This isn't arbitrary; it's how humans organize and understand knowledge. 

If these inherent language structures are so vital for expressing ideas, why do we discard them when building language models? 

## Introduction

Supermat inroduced a novel data representation framework for the AI era. 


## Citations

Citations present one of the most significant challenges in NLP: when an LLM answers a question, how do we trace its source information? This challenge intersects with important AI ethics discussions about properly attributing and recognizing others' work.

By preserving the inherent structure of documents, we propose a human and machine-readable citation system that addresses these concerns.

### Structure ID

The Structure ID is a unique referencing system that leverages language hierarchies (document, section, paragraph, and sentence) to precisely locate text. This intuitive system makes text highlighting and attribution straightforward.

The Strucutre ID, goes like this **`2.1.4.8`**. This points specifically to **Document Index number `2`**, the **`1`st section** in that document, then the **`4`th paragraph** in that section, and finally the **`8`th sentence** in that paragraph.
This simple yet powerful structure serves two purposes: it maintains connections between different text chunks while remaining token-efficient for LLM processing.

Through post-processing of LLM outputs, this structure enables direct retrieval of original content from source documents, reducing the need for token repetition and minimizing the risk of hallucinations.

## Evaluation

We chose to work with the [CUAD](https://www.atticusprojectai.org/cuad) dataset, a QnA dataset in the legal domain, to showcase our approach. The legal domain provides an excellent testing ground due to its precise and elaborate nature, making it one of the most challenging domains for LLMs due to document volume. Legal documents contain strong structural elements that are vital to preserve when performing NLP tasks.We evaluated our approach against two benchmarks:

1. A baseline standard chunking strategy
2. LangChain's current state-of-the-art [SemanticChunker](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html) with `breakpoint_threshold_type="percentile"`

In our internal evals, we see double-digit lifts in factual correctness and broader coverage with more complete outputs. This translates to fewer hallucinations and more trust in automated answers.

> **Accuracy: +15.56% | Faithfulness: +12.53% | ROUGE-1 Recall: +33.33%**

## Conclusion

Our initial findings suggest significant untapped potential in this area. This framework represents just the beginning of our work. We aim to demonstrate that preserving language structure in AI systems yields better results than traditional chunking methods used in RAG systems. Rather than discarding these structures due to complexity, we advocate for deeper investigation into how they can be preserved and leveraged effectively.
