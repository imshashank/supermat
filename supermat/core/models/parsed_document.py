# TODO (@legendof-selda): use alias for all attributes that are not pythonic
from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, Union
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
    _original_alias: dict = PrivateAttr(init=True)
    _unexisted_keys: set = PrivateAttr(init=True)

    def __init__(self, **data):
        aliases = {}
        unexisted_keys = set()
        for field_name, field in self.model_fields.items():
            alias_found = False
            if isinstance(field.validation_alias, AliasChoices):
                for alias in field.validation_alias.choices:
                    if alias in data:
                        aliases[field_name] = alias
                        alias_found = True
                        break
            elif field.alias is not None:
                aliases[field_name] = field.alias
                alias_found = True

            if (alias_found and aliases[field_name] not in data) or (not alias_found and field_name not in data):
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
        }
        serialized.update(aliased_values)
        cleaned_serialized = {
            field_name: value for field_name, value in serialized.items() if field_name not in self._unexisted_keys
        }
        return cleaned_serialized


class BaseChunkProperty(CustomBaseModel):
    ObjectID: int = Field(validation_alias=AliasChoices("ObjectID", "ObjectId"))
    Bounds: tuple[float | int, float | int, float | int, float | int]
    Page: int
    Path: str
    attributes: dict[str, Any] | None = None


class FontProperties(CustomBaseModel):
    alt_family_name: str
    embedded: bool
    encoding: str
    family_name: str
    font_type: str
    italic: bool
    monospaced: bool
    name: str
    subset: bool
    weight: int


class TextChunkProperty(BaseChunkProperty):
    Font: FontProperties
    HasClip: bool
    Lang: str | None = None
    TextSize: float | int


ChunkModelType: TypeAlias = Annotated[Union["TextChunk", "ImageChunk", "FootnoteChunk"], Field(discriminator="type_")]


class BaseChunk(CustomBaseModel):
    type_: Literal["Text", "Image", "Footnote"] = Field(alias="type")
    structure: str


class BaseTextChunk(BaseChunk):
    text: str
    key: list[str]
    properties: BaseChunkProperty
    sentences: list[ChunkModelType] | None = None


class TextChunk(BaseTextChunk):
    type_: Literal["Text"] = Field("Text", alias="type")
    speaker: dict | None = None
    document: str | None = None
    timestamp: str | None = None
    annotations: list[str] | None = None
    properties: TextChunkProperty


class ImageChunk(BaseChunk, BaseChunkProperty):
    type_: Literal["Image"] = Field("Image", alias="type")
    figure: str
    figure_object: Base64Bytes | None = Field(alias="figure-object", repr=False)

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
    type_: Literal["Footnote"] = Field("Footnote", alias="type")


ParsedDocumentType: TypeAlias = list[ChunkModelType]
ParsedDocument = TypeAdapter(ParsedDocumentType)


def load_parsed_document(path: Path | str) -> ParsedDocumentType:
    path = Path(path)
    with path.open("rb") as fp:
        raw_doc = orjson.loads(fp.read())

    if isinstance(raw_doc, list):
        return ParsedDocument.validate_python(raw_doc)
    elif isinstance(raw_doc, dict) and len(raw_doc.keys()) == 1:
        root_key = next(iter(raw_doc.keys()))
        warn(f"The json document contains a root node {next(iter(raw_doc.keys()))}.", ValidationWarning)
        return ParsedDocument.validate_python(raw_doc[root_key])
    else:
        raise orjson.JSONDecodeError("Invalid JSON Format")


def export_parsed_document(document: ParsedDocumentType, output_path: Path | str, **kwargs):
    output_path = Path(output_path)
    with output_path.open("wb+") as fp:
        fp.write(ParsedDocument.dump_json(document, **kwargs))
