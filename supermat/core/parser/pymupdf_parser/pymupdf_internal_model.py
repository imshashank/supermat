from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


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
    bbox: List[float]


class Block(BaseModel):
    number: int
    type_: int = Field(alias="type")
    bbox: List[float]
    lines: Optional[List[Line]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    ext: Optional[str] = None
    colorspace: Optional[int] = None
    xres: Optional[int] = None
    yres: Optional[int] = None
    bpc: Optional[int] = None
    transform: Optional[List[float]] = None
    size: Optional[int] = None
    image: Optional[str] = None


class Page(BaseModel):
    number: int  # page_number
    rect: Rect
    text: str
    blocks: List[Block]


class PyMuPDFDocument(BaseModel):
    filename: str
    total_pages: int
    pages: List[Page]
