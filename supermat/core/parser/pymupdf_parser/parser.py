from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from supermat.core.models.parsed_document import (
    FontProperties,
    ImageChunk,
    ParsedDocument,
    ParsedDocumentType,
    TextChunk,
    TextChunkProperty,
)
from supermat.core.parser.base import Parser
from supermat.core.parser.file_processor import FileProcessor
from supermat.core.parser.pymupdf_parser.pymupdf_internal_model import (
    ImageBlock,
    PyMuPDFDocument,
    TextBlock,
)
from supermat.core.parser.pymupdf_parser.utils import parse_pdf
from supermat.core.parser.utils import get_keywords
from supermat.core.utils import get_structure


def get_path(*args: int) -> str:
    """Create path from page number, block number and line number."""
    return "/".join(map(str, args))


def process_pymupdf(parsed_pdf: PyMuPDFDocument) -> ParsedDocumentType:
    """Converts a pdf, page and by page, and block by block using PyMuPDF.

    Args:
        parsed_pdf (PyMuPDFDocument): Pydantic model representation of pymupdf document.

    Returns:
        ParsedDocumentType: Parsed form of the pdf.
    """
    chunks = []
    for page in parsed_pdf.pages:
        for block in page.blocks:
            if isinstance(block, TextBlock):
                # TODO (@legendof-selda): create keys
                sentence_chunks: list[TextChunk] = []
                for line_no, line in enumerate(block.lines):
                    # NOTE: we are combining spans to lines as spans are not very relevant.
                    # All lines have a signle span
                    line_text = "".join(span.text for span in line.spans)
                    # NOTE: we are assuming the font properties to be the same for all spans in a line.
                    # This can lead to losing certain info like underline, bold, etc for small parts of texts.
                    first_span = line.spans[0]

                    structure = get_structure(page.number + 1, block.number + 1, line_no + 1)

                    sentence_chunk = TextChunk(
                        structure=structure,
                        properties=TextChunkProperty(
                            font=FontProperties(
                                name=first_span.font, **first_span.model_dump(exclude={"text", "bbox"})
                            ),
                            text_size=first_span.size,
                            bounds=line.bbox,
                            page=page.number,
                            path=get_path(page.number, block.number, line_no),
                            # TODO (@legendof-selda): need to figure out a way to get attributes if possible
                        ),
                        text=line_text,
                        key=get_keywords(line_text),
                    )
                    sentence_chunks.append(sentence_chunk)

                if TYPE_CHECKING:
                    assert sentence_chunks[0].properties

                text = " ".join(sentence_chunk.text for sentence_chunk in sentence_chunks)
                chunk = TextChunk(
                    structure=get_structure(page.number + 1, block.number + 1),
                    text=text,
                    key=get_keywords(text),
                    sentences=sentence_chunks if len(sentence_chunks) > 1 else None,
                    properties=(
                        None
                        if len(sentence_chunks) <= 1
                        else TextChunkProperty(
                            **(
                                sentence_chunks[0].properties.model_dump(exclude={"Path", "Bounds"})
                                | {"Path": get_path(page.number, block.number), "Bounds": block.bbox}
                            )
                        )
                    ),
                )
            elif isinstance(block, ImageBlock):
                chunk = ImageChunk(
                    structure=get_structure(page.number + 1, block.number + 1),
                    bounds=block.bbox,
                    page=page.number,
                    path=get_path(page.number, block.number),
                    figure_object=block.image,
                    attributes=block.model_dump(exclude={"number", "type_", "bbox", "image"}),
                )
            else:
                # NOTE: Need to figure out how to get Footnote type
                raise ValueError(f"Invalid block type {block.type_}")

            chunks.append(chunk)

    return ParsedDocument.validate_python(chunks)


@FileProcessor.register(".pdf")
class PyMuPDFParser(Parser):
    """Parses a pdf file using PyMuPDF library."""

    def parse(self, file_path: Path) -> ParsedDocumentType:
        parsed_pdf = parse_pdf(file_path)
        return process_pymupdf(parsed_pdf)
