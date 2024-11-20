from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List

import pymupdf

from supermat.core.models.parsed_document import (
    ChunkModelType,
    FontProperties,
    FootnoteChunk,
    ImageChunk,
    ParsedDocument,
    ParsedDocumentType,
    TextChunk,
    TextChunkProperty,
)
from supermat.core.parser.base import Parser
from supermat.core.parser.file_processor import FileProcessor


def get_font_properties(doc: pymupdf.Document, font_str: str) -> FontProperties:
    """Extract font properties from a PyMuPDF font string and document."""
    # Create default font properties
    font_props = FontProperties(
        alt_family_name="",
        embedded=False,
        encoding="",
        family_name="",
        font_type="",
        italic=False,
        monospaced=False,
        name=font_str,  # Use the font string as name
        subset=False,
        weight=400,
    )

    try:
        # Try to get more detailed font info from the document
        font_info = doc.get_font_info([font_str])
        if font_info and len(font_info) > 0:
            info = font_info[0]
            font_props = FontProperties(
                alt_family_name=info.get("family", ""),
                embedded=info.get("embedded", False),
                encoding=info.get("encoding", ""),
                family_name=info.get("family", ""),
                font_type=info.get("type", ""),
                italic=info.get("italic", False),
                monospaced=False,  # PyMuPDF doesn't provide this info directly
                name=info.get("name", font_str),
                subset=info.get("subset", False),
                weight=info.get("weight", 400),
            )
    except Exception:
        # If anything goes wrong, return the default properties
        pass

    return font_props


def extract_text_properties(doc: pymupdf.Document, span_dict: Dict[str, Any], page_num: int) -> TextChunkProperty:
    """Extract text properties from a PyMuPDF span dictionary."""
    return TextChunkProperty(
        ObjectID=0,  # PyMuPDF doesn't provide this directly
        Bounds=(span_dict.get("bbox")[0], span_dict.get("bbox")[1], span_dict.get("bbox")[2], span_dict.get("bbox")[3]),
        Page=page_num,
        Path=f"/page[{page_num}]/text[{span_dict.get('origin', (0, 0))[0]}]",
        Font=get_font_properties(doc, span_dict.get("font", "")),
        HasClip=False,  # PyMuPDF doesn't provide this directly
        Lang=None,  # PyMuPDF doesn't provide this directly
        TextSize=span_dict.get("size", 0),
        attributes={},
    )


def convert_image_to_base64(pix: pymupdf.Pixmap) -> bytes:
    """Convert PyMuPDF pixmap to base64 bytes."""
    img_bytes = pix.tobytes()
    return base64.b64encode(img_bytes)


def pdf_to_parsed_document(pdf_path: str) -> ParsedDocumentType:
    """Convert PDF file to ParsedDocumentType."""
    doc = pymupdf.open(pdf_path)
    parsed_chunks: List[ChunkModelType] = []

    for page_num, page in enumerate(doc):
        # Extract text chunks
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_chunk = TextChunk(
                            structure="paragraph",
                            text=span.get("text", ""),
                            key=[],
                            properties=extract_text_properties(doc, span, page_num),
                            sentences=None,
                            speaker=None,
                            document=None,
                            timestamp=None,
                            annotations=None,
                        )
                        parsed_chunks.append(text_chunk)

        # Extract images
        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)

            if base_image:
                pix = pymupdf.Pixmap(doc, xref)
                image_rect = page.get_image_bbox(img)

                image_chunk = ImageChunk(
                    structure="figure",
                    figure=f"image_{page_num}_{image_index}",
                    figure_object=convert_image_to_base64(pix),
                    ObjectID=xref,
                    Bounds=(image_rect[0], image_rect[1], image_rect[2], image_rect[3]),
                    Page=page_num,
                    Path=f"/page[{page_num}]/image[{image_index}]",
                    attributes=None,
                )
                parsed_chunks.append(image_chunk)

        # NOTE: Extract footnotes (simplified - might need to adjust the detection logic)
        footnote_blocks = [
            block
            for block in text_blocks
            if block.get("type") == 0
            and any(
                span.get("text", "").startswith(("*", "†", "‡"))
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
        ]

        for block in footnote_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    footnote_chunk = FootnoteChunk(
                        type="Footnote",
                        structure="footnote",
                        text=span.get("text", ""),
                        # TODO (@legendof-selda): integrate nltk to get keys.
                        # Make separate utility as this is common to all parsers
                        key=[],
                        properties=extract_text_properties(doc, span, page_num),
                        sentences=None,
                        speaker=None,
                        document=None,
                        timestamp=None,
                        annotations=None,
                    )
                    parsed_chunks.append(footnote_chunk)

    doc.close()
    return ParsedDocument.validate_python(parsed_chunks)


@FileProcessor.register(".pdf")
class PyMuPDFParser(Parser):
    def parse(self, file_path: Path) -> ParsedDocumentType:
        return pdf_to_parsed_document(file_path)
