# ![supermat](docs/assets/supermat-logo-black-sub.png "supermat")


## Preface

Current retrieval systems face two fundamental limitations that we've accepted as normal. 

First, as they fragment information during processing, they lose natural relationships. 
While vector search is powerful for finding semantically related content, similarity isn't same as actual relationships. Even the most sophisticated similarity search can't fully reconstruct explicit relationships nor make implicit connections clear . Systems end up spending massive resources trying to approximate context through similarity and further post-processing that was clear and direct before processing. 
The result:  increasingly sophisticated systems bogged down by trying to reconstruct what was there all along. 

Secondly, for the purpose of referencing, these systems use flat IDs - UUIDs and random strings - that can't express relationships, forcing them to maintain separate layers just to understand how information connects. Citations are an after-thought today.  

Our aproach solves both these problems with a fundamental insight. 
Information has natural connections - from documents to sections to paragraphs to sentences. This isn't arbitrary; it's how humans organize and understand knowledge. So why do we let AI systems break them apart? 

## Introduction

Supermat inroduces a novel data representation framework for the AI era, making relationships explicit and retrievable by design.


### Structured Citations

Our Structure ID is a unique referencing system that leverages language hierarchies (document, section, paragraph, and sentence) to precisely locate text. This intuitive system makes precise attribution straightforward.

The Structure ID, goes like this **`2.1.4.8`**. This points specifically to **Document Index number `2`**, the **`1`st section** in that document, then the **`4`th paragraph** in that section, and finally the **`8`th sentence** in that paragraph.
This simple yet powerful structure serves two purposes: it maintains connections between different text chunks while remaining token-efficient for LLM processing.

Through post-processing of LLM outputs, this structure enables direct retrieval of original content from source documents, reducing the need for token repetition and minimizing the risk of hallucinations.

## Evaluation

We chose to work with the [CUAD](https://www.atticusprojectai.org/cuad) dataset to showcase our approach. The legal domain provides an excellent testing ground due to its complexity and ambigious language. If Supermat works here, it can be generalized to other domains. We evaluated our approach against two benchmarks:

1. A baseline standard chunking strategy
2. LangChain's current state-of-the-art [SemanticChunker](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html) with `breakpoint_threshold_type="percentile"`

Key Metrics:

> **Accuracy: +15.56% | Faithfulness: +12.53% | ROUGE-1 Recall: +33.33%**

In our internal evals, we see double-digit lifts in factual correctness and broader coverage with more complete outputs. This translates to fewer hallucinations and more trust in automated answers.


## Conclusion

This is only the beginning of our journey as we build products on this foundation. 

Our findings suggest significant untapped potential in this area. Preserving and representing both explicit and implicit connections within data before any chunking or processing yields better results than immediatly jumping into chunking.

Far beyond just efficiency gains, we beleive this crucial intersection enables simplified yet powerful human + AI interfaces and novel operating models of tomorrow. We're building for those. 

## Contributing 

We welcome contributers and collaborators to join us on this journey. 
