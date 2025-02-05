# ![supermat](docs/assets/supermat-logo-black-sub.png "supermat")

## Preface

Supermat is a new way of thinking retrieval for AI. Language consists of semantic hierarchical structures to express ideas and information.

Writing isn't just about tokens/words, but on structure as well to express human thought. Documents, sections, paragraphs, sentences and words are the inherent structure that we find in all text. Just focusing on words might not be the best way to express ourselves, organizing concepts in the right structure can make it easier for humans to comprehend.
What makes a great text book? It isn't the words that it uses but how it organizes the different concepts in a certain order of sections and paragraphs that optimizes a person's understanding the best possible way.

These inherent structures in language help shape content and is one of the best ways to express ideas, then why do we throw it away while building language models. That is the question which we seek.

## Introduction

This repository explores the idea of keeping the inherent structures in language available in Large Language Models. We specifically targetted RAG systems to showcase this idea and demonstrate how effective it can be as compared to basic chunking and SOTA proposals from langchain.

Here, we specifically worked on the [CUAD](https://www.atticusprojectai.org/cuad) dataset, and QnA dataset in the legal domain, to demonstrate this idea. The legal domain is a great way to showcase this since they tend to be very precise and elaborate and is one of the most difficult domains for LLMs to work on due to it's large volume. Legal documents has a very strong structure in them, which is vital to keep when working with NLP based tasks.

## Evaluation

We compared our approach with standard chunking strategy as a baseline and the current SOTA chunking strategy available in langchain called [SemanticChunker](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html), with `breakpoint_threshold_type="percentile"`.

In our evaluation exercise, we found the following results.

> TODO: after discussing with Rishi, will figure out the best way to display this.

## Conclusion

From this, it is fair to say that there is a lot to be explored here. Our endevour has only begun. What we hope to gain here, is to show that focusing more on keeping the structure of language intact only rewards us more. Rather than going with the usual practices of chunking in most RAG systems, a new way of saving text as writing can lead to the next step. Instead of throwing away these structures for the reason of complexity, we would like to deep dive in and see more on how this can be saved.
