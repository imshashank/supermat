import base64
import os
import re
import zipfile
from pathlib import Path
from warnings import warn

from adobe.pdfservices.operation.auth.service_principal_credentials import (
    ServicePrincipalCredentials,
)
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
    ExtractElementType,
)
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
    ExtractPDFParams,
)
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import (
    ExtractRenditionsElementType,
)
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
    ExtractPDFResult,
)
from dotenv import find_dotenv, load_dotenv

from supermat.core.models.parsed_document import (
    FontProperties,
    FootnoteChunk,
    ImageChunk,
    ParsedDocument,
    ParsedDocumentType,
    TextChunk,
    TextChunkProperty,
)
from supermat.core.parser.adobe_parser._adobe_doc_cacher import CachedFile
from supermat.core.parser.adobe_parser.adobe_internal_model import (
    AdobeStructuredData,
    Element,
)
from supermat.core.parser.utils import get_keywords, split_text_into_token_chunks
from supermat.core.utils import get_structure, split_structure

load_dotenv(find_dotenv())

PDF_SERVICES_CLIENT_ID = os.environ.get("PDF_SERVICES_CLIENT_ID")
PDF_SERVICES_CLIENT_SECRET = os.environ.get("PDF_SERVICES_CLIENT_SECRET")
MAX_PARAGRAPH_LEN = int(os.environ.get("MAX_PARAGRAPH_LEN", 4000))
MIN_SENTENCE_LEN = int(os.environ.get("MIN_SENTENCE_LEN", 2))

CACHED_FILE = CachedFile()


def split_path(path: str) -> list[str]:
    return path.removeprefix("//").split("/")


def create_image_chunk(
    file_path: str, element: Element, archive: zipfile.ZipFile, element_structure: str, document_name: str
) -> ImageChunk | None:
    try:
        with archive.open(file_path) as f:
            file_data = base64.b64encode(f.read())

        assert element.Bounds is not None
        assert element.Page is not None
        image_chunk = ImageChunk(
            document=document_name,
            structure=element_structure,
            object_id=element.ObjectID,
            bounds=element.Bounds,
            page=element.Page,
            path=file_path,
            figure=file_path,
            figure_object=file_data,
        )
        return image_chunk
    except FileNotFoundError:
        warn(f"Warning: File not found - {file_path}", ResourceWarning)
        return None


def create_text_properties(element: Element) -> TextChunkProperty:
    assert element.Bounds is not None
    assert element.Page is not None
    return TextChunkProperty(
        object_id=element.ObjectID,
        bounds=element.Bounds,
        page=element.Page,
        path=element.Path,
        font=FontProperties(
            name=element.Font.name if element.Font else "Unknown",
            family_name=element.Font.family_name if element.Font else None,
            alt_family_name=element.Font.alt_family_name if element.Font else None,
            font_type=element.Font.font_type if element.Font else None,
            weight=element.Font.weight if element.Font else None,
            italic=element.Font.italic if element.Font else None,
            monospaced=element.Font.monospaced if element.Font else None,
            embedded=element.Font.embedded if element.Font else None,
            encoding=element.Font.encoding if element.Font else None,
            subset=element.Font.subset if element.Font else None,
        ),
        text_size=element.TextSize or 0,
        lang=element.Lang,
        hasclip=element.HasClip,
    )


def _create_sentence(sentence_structure: str, sentence: str, paragraph_chunk: TextChunk) -> TextChunk:
    copy_dict = paragraph_chunk.model_dump(include={"key", "properties"}, exclude_unset=True) | {
        "structure": sentence_structure,
        "text": sentence,
        "key": get_keywords(sentence),
    }
    return TextChunk(**copy_dict)


def append_sentences(text_chunk: TextChunk) -> TextChunk:
    # NOTE: This pattern works for unicode (non english) characters as well.
    setence_pattern = r"(?<=[^A-Z].[.?!])\s+(?=[^a-z])"
    # NOTE: sometimes some rogue substrings are found and we want to discard them.
    sentences = [
        sentence.strip()
        for sentence in re.split(setence_pattern, text_chunk.text)
        if len(sentence.strip()) > MIN_SENTENCE_LEN
    ]
    if not sentences or len(sentences) == 1:
        return text_chunk
    section_parts = split_structure(text_chunk.structure)[:-1]
    sentence_chunks = [
        _create_sentence(get_structure(*section_parts, sentence_number + 1), sentence, text_chunk)
        for sentence_number, sentence in enumerate(sentences)
    ]
    text_chunk.sentences = sentence_chunks
    # TODO (@legendof-selda): workaround. need to get this working correctly.
    text_chunk._unexisted_keys.remove("sentences")
    return text_chunk


def create_text_chunk(element: Element, element_structure: str, document_name: str) -> TextChunk | FootnoteChunk:
    assert element.Text
    if element.Path and element.Path.startswith("//Document/Footnote"):
        chunk = FootnoteChunk(
            document=document_name,
            structure=element_structure,
            text=element.Text,
            key=[],
            properties=create_text_properties(element),
        )
    else:
        chunk = TextChunk(
            document=document_name,
            structure=element_structure,
            text=element.Text,
            key=[],
            properties=create_text_properties(element),
        )
        chunk = append_sentences(chunk)
    chunk.key = get_keywords(chunk.text)
    return chunk


def process_list_items(
    elements: list[Element], starting_element_number: int, element_structure: str, document_name: str
) -> tuple[TextChunk | None, int]:
    next_elements = elements[starting_element_number:]
    if not next_elements:
        return None, 0

    all_list_item_text = []
    element_number = 0
    path = split_path(next_elements[0].Path)
    current_list_path = path[1]
    for element_number, element in enumerate(next_elements):
        path = split_path(element.Path)
        if path[1] != current_list_path:
            break
        if element.Text:
            all_list_item_text.append(element.Text)

    list_chunk_text = "\n".join(all_list_item_text)
    list_chunk = TextChunk(
        document=document_name,
        structure=element_structure,
        text=list_chunk_text,
        key=get_keywords(list_chunk_text),
        properties=create_text_properties(next_elements[0]),
    )
    list_chunk = append_sentences(list_chunk)
    return list_chunk, element_number - 1


def split_element_chunk(element: Element) -> list[Element]:
    assert element.Text

    chunks = split_text_into_token_chunks(element.Text, max_tokens=MAX_PARAGRAPH_LEN)

    if len(chunks) == 1:
        return [element]

    split_elements = [Element(**(element.model_dump(exclude={"Text"}) | {"Text": chunk})) for chunk in chunks]
    return split_elements


def convert_adobe_to_parsed_document(
    adobe_data: AdobeStructuredData,
    archive: zipfile.ZipFile,
    document_name: str,
) -> ParsedDocumentType:
    section_number = 0
    passage_number = 0
    figure_count = 0
    chunks: list[TextChunk | ImageChunk | FootnoteChunk] = []

    # TODO (@legendof-selda): not dealing with tables for now.
    skip_elements = 0
    for element_number, element in enumerate(adobe_data.elements):
        if skip_elements:
            skip_elements -= 1
            continue

        path = split_path(element.Path)
        if path[1][0] == "L" and path[2].startswith("LI") and path[-1] == "Lbl":
            # "//Document/L*/LI/Lbl"
            passage_number += 1
        elif path[1][0] == "H":
            # "//Document/H*
            section_number += 1
            passage_number = 0
        elif path[1][0] == "P":
            # "//Document/P*
            passage_number += 1
        elif path[1] == "Title":
            section_number += 1
            passage_number = 0
        else:
            passage_number += 1
        # TODO Collapse lists into a single paragraph and list items will sentence. collapse paragraphs as well+

        element_structure = get_structure(section_number, passage_number)

        if element.filePaths:
            for file_path in element.filePaths:
                image_chunk = create_image_chunk(file_path, element, archive, element_structure, document_name)
                if image_chunk:
                    figure_count += 1
                    image_chunk.figure = f"{figure_count} - {Path(file_path).name}"
                    chunks.append(image_chunk)
        elif path[1][0] == "L":
            chunk, skip_elements = process_list_items(
                adobe_data.elements, element_number, element_structure, document_name
            )
            if chunk:
                chunks.append(chunk)
        elif element.Text is not None and not (path[1].startswith("Table") or element.Bounds is None):
            element_chunks = split_element_chunk(element)
            for element_chunk in element_chunks:
                element_chunk_structure = get_structure(section_number, passage_number)
                chunk = create_text_chunk(element_chunk, element_chunk_structure, document_name)
                chunks.append(chunk)
                passage_number += 1

    return ParsedDocument.validate_python(chunks)


def load_adobe_zip(pdf_file: Path, zip_file: Path) -> ParsedDocumentType:
    with zipfile.ZipFile(zip_file, "r") as archive:
        structured_data_file = archive.open("structuredData.json")
        structured_data = AdobeStructuredData.model_validate_json(structured_data_file.read())
        documents = convert_adobe_to_parsed_document(structured_data, archive, document_name=pdf_file.stem)

        return documents


def adobe_parse(pdf_file: Path) -> Path:
    if CACHED_FILE.exists(pdf_file):
        cached_path = CACHED_FILE.get_cached_file_path(pdf_file)
        print(f"'{pdf_file}' is cached. Returning '{cached_path}'")
        return cached_path

    with pdf_file.open("rb") as fp:
        input_stream = fp.read()

    # NOTE: code extracted from
    # https://developer.adobe.com/document-services/docs/overview/pdf-extract-api/quickstarts/python/
    credentials = ServicePrincipalCredentials(
        client_id=PDF_SERVICES_CLIENT_ID, client_secret=PDF_SERVICES_CLIENT_SECRET
    )

    pdf_services = PDFServices(credentials=credentials)

    input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

    extract_pdf_params = ExtractPDFParams(
        elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
        elements_to_extract_renditions=[ExtractRenditionsElementType.TABLES, ExtractRenditionsElementType.FIGURES],
    )

    extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

    location = pdf_services.submit(extract_pdf_job)
    pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

    # Get content from the resulting asset(s)
    result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
    stream_asset: StreamAsset = pdf_services.get_content(result_asset)

    zip_file = CACHED_FILE.create_file(pdf_file)
    # Creates an output stream and copy stream asset's content to it
    with zip_file.open(mode="wb+") as fp:
        fp.write(stream_asset.get_input_stream())

    return zip_file


def parse_file(pdf_file: Path) -> ParsedDocumentType:
    zip_file = adobe_parse(pdf_file)
    return load_adobe_zip(pdf_file, zip_file)
