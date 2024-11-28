import os
import zipfile
from pathlib import Path

import orjson
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

from supermat.core.models.parsed_document import ParsedDocument, ParsedDocumentType
from supermat.core.parser.adobe_parser._utils import CachedFile

load_dotenv(find_dotenv())

PDF_SERVICES_CLIENT_ID = os.environ.get("PDF_SERVICES_CLIENT_ID")
PDF_SERVICES_CLIENT_SECRET = os.environ.get("PDF_SERVICES_CLIENT_SECRET")
CACHED_FILE = CachedFile()


def load_adobe_zip(zip_file: Path) -> ParsedDocumentType:
    with zipfile.ZipFile(zip_file, "r") as archive:
        structured_data_file = archive.open("structuredData.json")
        structured_data = orjson.loads(structured_data_file.read())
        return ParsedDocument.validate_python(structured_data["elements"])


def adobe_parse(pdf_file: Path) -> Path:
    if CACHED_FILE.exists(pdf_file):
        cached_path = CACHED_FILE.get_cached_file_path(pdf_file)
        print(f"'{pdf_file}' is cached. Returning '{cached_path}'")
        return cached_path

    with pdf_file.open("rb") as fp:
        input_stream = fp.read()

    # Initial setup, create credentials instance
    credentials = ServicePrincipalCredentials(
        client_id=PDF_SERVICES_CLIENT_ID, client_secret=PDF_SERVICES_CLIENT_SECRET
    )

    # Creates a PDF Services instance
    pdf_services = PDFServices(credentials=credentials)

    # Creates an asset(s) from source file(s) and upload
    input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

    # Create parameters for the job
    extract_pdf_params = ExtractPDFParams(
        elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
        elements_to_extract_renditions=[ExtractRenditionsElementType.TABLES, ExtractRenditionsElementType.FIGURES],
    )

    # Creates a new job instance
    extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

    # Submit the job and gets the job result
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
    return load_adobe_zip(zip_file)
