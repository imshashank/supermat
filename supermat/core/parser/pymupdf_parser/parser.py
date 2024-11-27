from __future__ import annotations

from pathlib import Path

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
from supermat.core.parser.pymupdf_parser.pymupdf_internal_model import PyMuPDFDocument
from supermat.core.parser.pymupdf_parser.utils import parse_pdf


def get_structure(*args: int, min_length: int = 3) -> str:
    if len(args) < min_length:
        args = args + (-1,) * (min_length - len(args))
    return ".".join(map(lambda x: str(x + 1), args))


def get_path(*args: int) -> str:
    return "/".join(map(str, args))


def process_pymupdf(parsed_pdf: PyMuPDFDocument) -> ParsedDocumentType:
    chunks = []
    for page in parsed_pdf.pages:
        for block in page.blocks:
            if block.type_ == 0:
                # TODO (@legendof-selda): create keys
                sentence_chunks: list[TextChunk] = []
                for line_no, line in enumerate(block.lines):
                    # NOTE: we are combining spans to lines as spans are not very relevant.
                    # All lines have a signle span
                    line_text = "".join(span.text for span in line.spans)
                    # NOTE: we are assuming the font properties to be the same for all spans in a line.
                    # This can lead to losing certain info like underline, bold, etc for small parts of texts.
                    first_span = line.spans[0]

                    structure = get_structure(page.number, block.number, line_no)

                    sentence_chunk = TextChunk(
                        structure=structure,
                        properties=TextChunkProperty(
                            font=FontProperties(
                                name=first_span.font, **first_span.model_dump(exclude=["text", "bbox"])
                            ),
                            text_size=first_span.size,
                            bounds=tuple(line.bbox),
                            page=page.number,
                            path=get_path(page.number, block.number, line_no),
                            # TODO (@legendof-selda): need to figure out a way to get attributes if possible
                        ),
                        text=line_text,
                        key=[],
                    )
                    sentence_chunks.append(sentence_chunk)

                chunk = TextChunk(
                    structure=get_structure(page.number, block.number),
                    text=" ".join(sentence_chunk.text for sentence_chunk in sentence_chunks),
                    key=[],
                    sentences=sentence_chunks if len(sentence_chunks) > 1 else None,
                    properties=(
                        None
                        if len(sentence_chunks) <= 1
                        else (
                            sentence_chunks[0].properties.model_dump(exclude=["Path", "Bounds"])
                            | {"Path": get_path(page.number, block.number), "Bounds": block.bbox}
                        )
                    ),
                )
            elif block.type_ == 1:
                assert block.image is not None
                chunk = ImageChunk(
                    structure=get_structure(page.number, block.number),
                    bounds=tuple(block.bbox),
                    page=page.number,
                    path=get_path(page.number, block.number),
                    figure_object=block.image,
                    attributes=block.model_dump(exclude=["number", "type_", "bbox", "image"]),
                )
            else:
                # NOTE: Need to figure out how to get Footnote type
                raise ValueError(f"Invalid block type {block.type_}")

            chunks.append(chunk)

    return ParsedDocument.validate_python(chunks)


@FileProcessor.register(".pdf")
class PyMuPDFParser(Parser):
    def parse(self, file_path: Path) -> ParsedDocumentType:
        parsed_pdf = parse_pdf(file_path)
        return process_pymupdf(parsed_pdf)
