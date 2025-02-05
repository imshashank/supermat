"""
The parser submodule contains all Parser implementation that converts a given file type to a ParsedDocument.
For the Parser to be registered, it needs to be included here.
TODO (@legendof-selda): Dynamically register all parsers.

To create a new `Parser`, create a submodule for it and inside the submodule, it should have `parser.py`.
Here is where the `Parser` implementation will be written.
For any utilities associated to that parser will go to `utils.py`.
Also include import the Parser in its corresponding `__init__.py` file for easier importing.
"""

from supermat.core.parser.adobe_parser import AdobeParser
from supermat.core.parser.file_processor import FileProcessor
from supermat.core.parser.pymupdf_parser import PyMuPDFParser

__all__ = [
    "AdobeParser",
    "FileProcessor",
    "PyMuPDFParser",
]
