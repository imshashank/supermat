from pathlib import Path

from supermat.core.models.parsed_document import ParsedDocumentType
from supermat.core.parser.adobe_parser.parser import parse_file
from supermat.core.parser.base import Parser
from supermat.core.parser.file_processor import FileProcessor


@FileProcessor.register(".pdf", main=True)
class AdobeParser(Parser):
    def parse(self, file_path: Path) -> ParsedDocumentType:
        return parse_file(file_path)


__all__ = ["AdobeParser"]
