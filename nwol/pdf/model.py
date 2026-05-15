from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_seq(cls, values: Any) -> "BBox":
        if isinstance(values, BBox):
            return values
        if not isinstance(values, (list, tuple)) or len(values) < 4:
            return cls(0.0, 0.0, 0.0, 0.0)
        return cls(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def union(self, other: "BBox | None") -> "BBox":
        if other is None:
            return self
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )


BoundingBox = BBox


@dataclass(slots=True)
class RawSpan:
    text: str
    bbox: BBox
    page: int
    font_size: float | None = None
    font_name: str | None = None
    is_bold: bool = False
    flags: int | None = None


@dataclass(slots=True)
class RawLine:
    text: str
    bbox: BBox
    page: int
    font_size: float | None = None
    font_name: str | None = None
    is_bold: bool = False
    spans: list[RawSpan] = field(default_factory=list)


@dataclass(slots=True)
class RawBlock:
    text: str
    bbox: BBox
    page: int
    block_type: str = "unknown"
    lines: list[RawLine] = field(default_factory=list)

    @property
    def font_size(self) -> float | None:
        sizes = [line.font_size for line in self.lines if line.font_size]
        if not sizes:
            return None
        return sum(sizes) / len(sizes)

    @property
    def is_bold(self) -> bool:
        return any(line.is_bold for line in self.lines)


@dataclass(slots=True)
class RawPage:
    number: int
    width: float
    height: float
    lines: list[RawLine] = field(default_factory=list)


@dataclass(slots=True)
class RawDocument:
    pages: list[RawPage]
    path: str | None = None
    engine_name: str = "pymupdf_structured"
    warnings: list[str] = field(default_factory=list)

    @property
    def page_sizes(self) -> dict[int, tuple[float, float]]:
        return {page.number: (page.width, page.height) for page in self.pages}

    @property
    def lines(self) -> list[RawLine]:
        return [line for page in self.pages for line in page.lines]

    def replace_lines(self, lines: list[RawLine], warnings: list[str] | None = None) -> "RawDocument":
        by_page: dict[int, list[RawLine]] = {page.number: [] for page in self.pages}
        for line in lines:
            by_page.setdefault(line.page, []).append(line)
        pages = [
            RawPage(page.number, page.width, page.height, by_page.get(page.number, []))
            for page in self.pages
        ]
        return RawDocument(
            pages=pages,
            path=self.path,
            engine_name=self.engine_name,
            warnings=list(warnings if warnings is not None else self.warnings),
        )


@dataclass(slots=True)
class DocumentBlock:
    type: str
    text: str = ""
    page: int | None = None
    bbox: BBox | None = None
    level: int | None = None
    items: list[str] | None = None
    latex: str | None = None
    html: str | None = None
    markdown: str | None = None
    image_path: str | None = None
    caption: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def to_reader_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.id is not None:
            data["id"] = self.id

        if self.text:
            data["text"] = self.text
        elif self.type == "formula" and self.latex:
            data["text"] = self.latex
        elif self.type == "bullet_list" and self.items:
            data["text"] = "\n".join(f"• {item}" for item in self.items)
        else:
            data["text"] = self.text

        if self.page is not None:
            page_start = _coerce_positive_int((self.metadata or {}).get("page_start"), self.page)
            page_end = _coerce_positive_int((self.metadata or {}).get("page_end"), page_start)
            data["page_number"] = page_start
            data["page_start"] = page_start
            data["page_end"] = max(page_start, page_end)
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_list()
        if self.level is not None:
            data["level"] = self.level
        if self.items is not None:
            data["items"] = list(self.items)
        if self.latex is not None:
            data["latex"] = self.latex
            data.setdefault("display", True)
        if self.html is not None:
            data["html"] = self.html
        if self.markdown is not None:
            data["markdown"] = self.markdown
        if self.image_path is not None:
            data["image_path"] = self.image_path
        if self.caption is not None:
            data["caption"] = self.caption
            if not data.get("text"):
                data["text"] = self.caption
        data["confidence"] = float(max(0.0, min(1.0, self.confidence)))
        if self.metadata:
            data["metadata"] = dict(self.metadata)
            for key in ("is_caption", "is_metadata", "engine", "block_index", "caption_display", "caption_group"):
                if key in self.metadata:
                    data[key] = self.metadata[key]
        return data

    @classmethod
    def from_reader_dict(cls, block: dict[str, Any]) -> "DocumentBlock":
        bbox = BBox.from_seq(block["bbox"]) if block.get("bbox") is not None else None
        page = block.get("page") or block.get("page_number") or block.get("page_start")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        known = {
            "id",
            "type",
            "text",
            "page",
            "page_number",
            "page_start",
            "page_end",
            "bbox",
            "level",
            "items",
            "latex",
            "html",
            "markdown",
            "image_path",
            "caption",
            "confidence",
            "metadata",
        }
        metadata = dict(block.get("metadata") or {})
        if block.get("page_start") is not None:
            metadata.setdefault("page_start", block.get("page_start"))
        if block.get("page_end") is not None:
            metadata.setdefault("page_end", block.get("page_end"))
        metadata.update({key: value for key, value in block.items() if key not in known})
        return cls(
            type=str(block.get("type") or "paragraph"),
            text=str(block.get("text") or ""),
            page=page,
            bbox=bbox,
            level=block.get("level"),
            items=list(block.get("items") or []) if block.get("items") is not None else None,
            latex=block.get("latex"),
            html=block.get("html"),
            markdown=block.get("markdown"),
            image_path=block.get("image_path"),
            caption=block.get("caption"),
            confidence=float(block.get("confidence", 1.0) or 1.0),
            metadata=metadata,
            id=block.get("id"),
        )


@dataclass(slots=True)
class DocumentModel:
    blocks: list[DocumentBlock]
    pages: int
    score: float
    warnings: list[str]
    engine_name: str
    debug_paths: list[str] = field(default_factory=list)

    def to_reader_blocks(self) -> list[dict[str, Any]]:
        return [block.to_reader_dict() for block in self.blocks]


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(1, parsed)
