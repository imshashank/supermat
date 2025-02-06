"""
This is where all chunking strategies on ParsedDocuments are written.
Chunking strategies are strategies to best store the ParsedDocuments in a vector store or for LLM processing.
"""

from supermat.core.chunking.base import BaseChunker

__all__ = ["BaseChunker"]
