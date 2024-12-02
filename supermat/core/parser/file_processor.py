from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from supermat.core.models.parsed_document import export_parsed_document
from supermat.core.parser.base import Converter, Parser

if TYPE_CHECKING:
    from supermat.core.models.parsed_document import ParsedDocumentType


@dataclass
class Handler:
    parser: Parser
    converters: tuple[Converter, ...] | None = None

    @property
    def name(self) -> str:
        name = f"{type(self.parser).__name__}"
        if self.converters:
            name += f'[{"|".join((type(converter).__name__ for converter in self.converters))}]'
        return name

    def convert(self, file_path: Path) -> Path:
        return reduce(lambda r, f: f.convert(r), self.converters, file_path) if self.converters else file_path

    def parse(self, file_path: Path) -> ParsedDocumentType:
        return self.parser.parse(self.convert(file_path))

    def parse_file(self, file_path: Path | str) -> ParsedDocumentType:
        file_path = Path(file_path)
        parsed_document = self.parse(file_path)
        return parsed_document

    def process_file(self, file_path: Path | str, **kwargs) -> Path:
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        parsed_document = self.parse_file(file_path)
        parsed_out_file = file_path.with_suffix(f"{file_ext}.json")
        export_parsed_document(parsed_document, parsed_out_file, **kwargs)
        return parsed_out_file


class FileProcessor:
    _registered_handlers: dict[str, Handler] = {}
    _handlers: dict[str, list[str]] = defaultdict(list)
    _main_handlers: dict[str, str] = {}
    _file_extension_pattern = re.compile(r"^\.[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*$")

    @staticmethod
    def _register(handler: Handler, extension: str, main: bool):
        FileProcessor._registered_handlers[handler.name] = handler
        FileProcessor._handlers[extension].append(handler.name)
        if main:
            FileProcessor._main_handlers[extension] = handler.name

    @staticmethod
    def register(
        extension: str, *, converters: type[Converter] | Iterable[type[Converter]] | None = None, main: bool = False
    ):
        # NOTE: this only works if the register has reached. Meaning we need to manually import it in __init__.py
        extension = extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        if not FileProcessor._file_extension_pattern.match(extension):
            raise ValueError(f"Invalid file extension: {extension}")
        if converters is not None and not isinstance(converters, Iterable):
            converters = (converters,)
        if converters is not None and (
            not_converters := [converter for converter in converters if not issubclass(converter, Converter)]
        ):
            raise TypeError(f"{not_converters} are not subclasses of {Converter}")
        if main and extension in FileProcessor._main_handlers:
            raise ValueError(
                f"{extension} is already registered to {FileProcessor._main_handlers[extension]}! "
                "Only one main parser can be registered for given extension."
            )

        def decorator(parser: type[Parser]) -> type[Parser]:
            if not issubclass(parser, Parser):
                raise TypeError(f"{parser} is not a subclass of {Parser}")
            handler = Handler(
                parser=parser(), converters=tuple(converter() for converter in converters) if converters else None
            )
            FileProcessor._register(handler, extension, main=main)
            return parser

        return decorator

    @staticmethod
    def get_main_handler(file_path: Path | str) -> Handler:
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        handler_id = FileProcessor._main_handlers.get(file_ext, None)
        if handler_id is None:
            raise ValueError(f"No main handler registered for file type: {file_ext}")

        return FileProcessor._registered_handlers[handler_id]

    @staticmethod
    def get_handler(handler_name: str) -> Handler:
        return FileProcessor._registered_handlers[handler_name]

    @staticmethod
    def get_handlers(file_path: Path | str) -> dict[str, Handler]:
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        return {
            handle_name: FileProcessor.get_handler(handle_name)
            for handle_name in FileProcessor._handlers.get(file_ext, [])
        }

    @staticmethod
    def parse_file(file_path: Path | str) -> ParsedDocumentType:
        handler = FileProcessor.get_main_handler(file_path)
        return handler.parse_file(file_path)

    @staticmethod
    def process_file(file_path: Path | str, **kwargs) -> Path:
        handler = FileProcessor.get_main_handler(file_path)
        return handler.process_file(file_path, **kwargs)
