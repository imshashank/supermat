"""
Base abstractions of Parser and Converters.
`Parser` parses a given document type into a `ParsedDocumentType`.
`Converter` converts a given document from one format to another so that it can be compatible with an existing `Parser`.
Example: We have a `Parser` that parses a .pdf document, we can have `Converter`s that convert docx, pptx into pdf.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supermat.core.models.parsed_document import ParsedDocumentType


class Parser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocumentType:  # noqa: U100
        """
        Parse give file to ParsedDocumentType.

        Args:
            file_path (Path): Input file.

        Returns:
            ParsedDocumentType: Parsed document
        """


class Converter(ABC):
    @abstractmethod
    def convert(self, file_path: Path) -> Path:  # noqa: U100
        """
        Converts input file to another file type and saves it. The saved file path is returned.

        Args:
            file_path (Path): Input file.

        Returns:
            Path: Output file after conversion.
        """

    def __call__(self, file_path: Path) -> Path:
        return self.convert(file_path)
