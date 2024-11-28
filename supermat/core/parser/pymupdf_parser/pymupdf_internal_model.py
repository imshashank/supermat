from __future__ import annotations

from typing import Annotated, List, Literal

from pydantic import BaseModel, Field

from supermat.core.models.parsed_document import Base64Bytes


class Rect(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class Span(BaseModel):
    size: float
    flags: int
    font: str
    color: int
    ascender: float
    descender: float
    text: str
    origin: List[float]
    bbox: List[float]


class Line(BaseModel):
    spans: List[Span]
    wmode: int
    direction: List[float] = Field(alias="dir")
    bbox: tuple[float, float, float, float]


class Block(BaseModel):
    type_: Literal[0, 1] = Field(alias="type")
    number: int
    bbox: tuple[float, float, float, float]


class TextBlock(Block):
    type_: Literal[0] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default=0, alias="type", frozen=True
    )
    lines: list[Line]


class ImageBlock(Block):
    type_: Literal[1] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default=1, alias="type", frozen=True
    )
    width: int
    height: int
    ext: str
    colorspace: int
    xres: int
    yres: int
    bpc: int
    transform: list[float]
    size: int
    image: Base64Bytes


class Page(BaseModel):
    number: int  # page_number
    rect: Rect
    text: str
    blocks: list[Annotated[TextBlock | ImageBlock, Field(discriminator="type_")]]


class PyMuPDFDocument(BaseModel):
    filename: str
    total_pages: int
    pages: List[Page]
