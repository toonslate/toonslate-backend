"""파이프라인 데이터 모델

Detection → Translation → Inpainting → Rendering 전체에서 사용하는 공통 스키마
"""

import math
from typing import Self

from pydantic import BaseModel, model_validator


class BBox(BaseModel):
    """바운딩 박스 [x1, y1, x2, y2]

    유효성:
    - x1 <= x2, y1 <= y2 보장 (자동 정렬)
    - 모든 좌표는 0 이상
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_and_normalize(self) -> Self:
        """좌표 유효성 검증 및 정규화"""
        # x1 <= x2, y1 <= y2 보장 (역전된 경우 자동 정렬)
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

        # 음수 좌표는 0으로 클램핑
        self.x1 = max(0.0, self.x1)
        self.y1 = max(0.0, self.y1)
        self.x2 = max(0.0, self.x2)
        self.y2 = max(0.0, self.y2)

        return self

    @classmethod
    def from_list(cls, coords: list[float]) -> "BBox":
        """리스트에서 BBox 생성

        Args:
            coords: [x1, y1, x2, y2] 형태의 리스트

        Raises:
            ValueError: 좌표 개수가 4개가 아니거나 숫자가 아닌 경우
        """
        if len(coords) != 4:
            raise ValueError(f"BBox requires 4 coordinates, got {len(coords)}")

        for i, c in enumerate(coords):
            if math.isnan(c) or math.isinf(c):
                raise ValueError(f"Coordinate {i} is NaN or Inf")

        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def to_tuple(self) -> tuple[int, int, int, int]:
        """정수 튜플로 변환 (PIL crop 등에 사용)

        round()를 사용하여 반올림 (truncation 방지)
        """
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))

    def to_list(self) -> list[float]:
        """리스트로 변환"""
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """중심점 (cx, cy)"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def is_valid(self) -> bool:
        """유효한 영역인지 확인 (width > 0 and height > 0)"""
        return self.width > 0 and self.height > 0


class TextRegion(BaseModel):
    """텍스트 영역 (Detection 결과 + 연관된 bubble)

    index는 원본 detection 결과의 순서를 유지하며,
    translation 결과와 매칭할 때 사용됩니다.
    """

    index: int  # 원본 detection 인덱스 (translation 매칭용)
    text_bbox: BBox
    bubble_bbox: BBox | None = None  # 포함하는 말풍선 (없으면 나레이션)
    fill_bbox: BBox | None = None  # Inpainting용 확장된 영역
    render_bbox: BBox | None = None  # 렌더링용 안전 영역


class TranslationResult(BaseModel):
    """Gemini 번역 결과 (단일 영역)"""

    index: int
    translated: str


class TranslatedRegion(BaseModel):
    """번역된 텍스트 영역"""

    region: TextRegion
    original_text: str = ""
    translated_text: str = ""


class PipelineResult(BaseModel):
    """전체 파이프라인 결과"""

    job_id: str
    regions: list[TranslatedRegion]
    original_path: str
    clean_path: str  # Inpainting 후
    result_path: str  # Rendering 후
