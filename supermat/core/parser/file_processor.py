import re
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from supermat.core.models.parsed_document import export_parsed_document

if TYPE_CHECKING:
    from supermat.core.models.parsed_document import ParsedDocumentType
    from supermat.core.parser.base import Converter, Parser


@dataclass
class Handler:
    parser: Parser
    converters: tuple[Converter] | None = None

    def convert(self, file_path: Path) -> Path:
        return reduce(lambda r, f: f(r), self.converters, file_path) if self.converters else file_path

    def parse(self, file_path: Path) -> ParsedDocumentType:
        return self.parser.parse(self.convert(file_path))


class FileProcessor:
    _handlers: dict[str, Handler] = {}
    _file_extension_pattern = re.compile(r"^\.[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*$")

    @staticmethod
    def register(extension: str, converters: type[Converter] | Iterable[type[Converter]] | None = None):
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
        if extension in FileProcessor._handlers:
            raise ValueError(
                f"{extension} is already registered to {type(FileProcessor._handlers[extension].parser)}! "
                "Only one parser can be registered for given extension."
            )

        def decorator(parser: type[Parser]) -> type[Parser]:
            if not issubclass(parser, Parser):
                raise TypeError(f"{parser} is not a subclass of {Parser}")
            FileProcessor._handlers[extension] = Handler(
                parser=parser(), converters=tuple(converter() for converter in converters) if converters else None
            )
            return parser

        return decorator

    @staticmethod
    def process_file(file_path: Path | str, **kwargs) -> Path:
        """
        Process a file by finding the appropriate handler and delegating the task.
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix
        handler = FileProcessor._handlers.get(file_ext)
        if handler is None:
            raise ValueError(f"No handler registered for file type: {file_ext}")
        parsed_document = handler.parse(file_path)
        parsed_out_file = file_path.with_suffix(f"{file_ext}.json")
        export_parsed_document(parsed_document, parsed_out_file, **kwargs)
        return parsed_out_file
