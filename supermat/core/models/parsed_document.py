from __future__ import annotations

import base64
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Sequence,
    TypeAlias,
    Union,
    overload,
)
from warnings import warn

import orjson
from pydantic import (
    AliasChoices,
    Base64Encoder,
    BaseModel,
    ConfigDict,
    EncodedBytes,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_serializer,
)

from supermat.core.utils import is_subsection


class Base64EncoderSansNewline(Base64Encoder):
    @classmethod
    def encode(cls, value: bytes) -> bytes:
        return base64.b64encode(value)


# NOTE: we dont use pydantic Base64Bytes since it includes embedding newlines
# https://github.com/pydantic/pydantic/issues/9072
Base64Bytes = Annotated[bytes, EncodedBytes(encoder=Base64EncoderSansNewline)]


class ValidationWarning(UserWarning):
    """Custom warning for validation issues in Pydantic models."""


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"by_alias": True}, extra="forbid")
    _original_alias: dict[str, str] = PrivateAttr()
    _unexisted_keys: set[str] = PrivateAttr()

    def __init__(self, **data: dict[str, Any]):
        aliases: dict[str, str] = {}
        unexisted_keys: set[str] = set()
        for field_name, field in self.model_fields.items():
            alias_found = False
            if isinstance(field.validation_alias, AliasChoices):
                for alias in field.validation_alias.choices:
                    if alias in data:
                        aliases[field_name] = alias
                        alias_found = True
                        break
            elif field.alias is not None or field.validation_alias is not None:
                alias = field.alias or field.validation_alias
                if TYPE_CHECKING:
                    assert isinstance(alias, str)
                aliases[field_name] = alias
                alias_found = True

            if not ((alias_found and aliases[field_name] in data) or (field_name in data)):
                unexisted_keys.add(aliases[field_name] if alias_found else field_name)

        super().__init__(**data)
        self._original_alias = aliases
        self._unexisted_keys = unexisted_keys

    @model_serializer(mode="wrap")
    def serialize_model(self, nxt: SerializerFunctionWrapHandler) -> dict[str, Any]:
        serialized = nxt(self)
        aliased_values = {
            renamed_field_name: serialized.pop(field_name)
            for field_name, renamed_field_name in self._original_alias.items()
            if field_name in serialized
        }
        serialized.update(aliased_values)
        _unexisted_keys = self._unexisted_keys - {
            field.alias or field_name for field_name, field in self.model_fields.items() if field.frozen
        }
        cleaned_serialized = {
            field_name: value for field_name, value in serialized.items() if field_name not in _unexisted_keys
        }
        return cleaned_serialized


class BaseChunkProperty(CustomBaseModel):
    object_id: int | None = Field(default=None, validation_alias=AliasChoices("ObjectID", "ObjectId"))
    bounds: tuple[float | int, float | int, float | int, float | int] = Field(validation_alias="Bounds")
    page: int = Field(validation_alias="Page")
    path: str | None = Field(default=None, validation_alias="Path")
    attributes: dict[str, Any] | None = None


class FontProperties(CustomBaseModel):
    model_config = ConfigDict(extra="allow")
    alt_family_name: str | None = None
    embedded: bool | None = None
    encoding: str | None = None
    family_name: str | None = None
    font_type: str | None = None
    italic: bool | None = None
    monospaced: bool | None = None
    name: str
    subset: bool | None = None
    weight: int | None = None


class TextChunkProperty(BaseChunkProperty):
    font: FontProperties = Field(validation_alias="Font")
    hasclip: bool | None = Field(default=None, validation_alias="HasClip")
    lang: str | None = Field(default=None, validation_alias="Lang")
    text_size: float | int = Field(validation_alias="TextSize")


ChunkModelForwardRefType: TypeAlias = Annotated[
    Union["TextChunk", "ImageChunk", "FootnoteChunk"], Field(discriminator="type_")
]


class BaseChunk(CustomBaseModel):
    type_: Literal["Text", "Image", "Footnote"] = Field(alias="type", frozen=True)
    structure: str

    @overload
    def is_subsection(self, sub_section: BaseChunk) -> bool: ...  # noqa: U100, E704

    @overload
    def is_subsection(self, sub_section: str) -> bool: ...  # noqa: U100, E704

    def is_subsection(self, sub_section: BaseChunk | str) -> bool:
        return is_subsection(sub_section if isinstance(sub_section, str) else sub_section.structure, self.structure)


class BaseTextChunk(BaseChunk):
    text: str
    key: list[str]
    properties: BaseChunkProperty | None = None
    sentences: Sequence[ChunkModelForwardRefType] | None = None


class TextChunk(BaseTextChunk):
    type_: Literal["Text"] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default="Text", alias="type", frozen=True
    )
    speaker: dict[str, Any] | None = None
    document: str | None = None
    timestamp: str | None = None
    annotations: list[str] | None = None
    properties: TextChunkProperty | None = None  # pyright: ignore[reportIncompatibleVariableOverride]


class ImageChunk(BaseChunk, BaseChunkProperty):
    type_: Literal["Image"] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default="Image", alias="type", frozen=True
    )
    figure: str | None = None
    figure_object: Base64Bytes | None = Field(validation_alias="figure-object", repr=False)

    @field_validator("figure_object", mode="before")
    @classmethod
    def validate_data(cls, value: Base64Bytes | None, info: ValidationInfo):  # noqa: U100
        # TODO (@legendof-selda): figure out a way to find the path where this fails.
        # NOTE: This shouldn't be allowed, but in the sample we have a case where the images aren't saved.
        if value is None:
            warn(f"{info.field_name} is None.", ValidationWarning)
            return None
        return value


class FootnoteChunk(TextChunk):
    type_: Literal["Footnote"] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default="Footnote", alias="type", frozen=True
    )


# NOTE: had to do this again as there were so many issues with ForwardRefs
ChunkModelType: TypeAlias = Annotated[TextChunk | ImageChunk | FootnoteChunk, Field(discriminator="type_")]
ParsedDocumentType: TypeAlias = list[ChunkModelType]
ParsedDocument = TypeAdapter(ParsedDocumentType)


def load_parsed_document(path: Path | str) -> ParsedDocumentType:
    path = Path(path)
    with path.open("rb") as fp:
        raw_doc: list[dict[str, Any]] | dict[str, list[dict[str, Any]]] = orjson.loads(fp.read())

    if isinstance(raw_doc, dict) and len(raw_doc.keys()) == 1:
        root_key = next(iter(raw_doc.keys()))
        warn(f"The json document contains a root node {next(iter(raw_doc.keys()))}.", ValidationWarning)
        return ParsedDocument.validate_python(raw_doc[root_key])
    elif isinstance(raw_doc, list):
        return ParsedDocument.validate_python(raw_doc)
    else:
        raise ValueError("Invalid JSON Format")


def export_parsed_document(document: ParsedDocumentType, output_path: Path | str, **kwargs):
    output_path = Path(output_path)
    with output_path.open("wb+") as fp:
        fp.write(ParsedDocument.dump_json(document, **kwargs))
