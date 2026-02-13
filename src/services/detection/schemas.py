"""Detection 스키마"""

from pydantic import BaseModel


class ImageSize(BaseModel):
    width: int
    height: int


class DetectionResult(BaseModel):
    """탐지 결과

    모든 좌표는 원본 이미지 기준 절대 좌표(px).
    bubbles/texts 각 항목은 [x1, y1, x2, y2] 형태.
    """

    image_size: ImageSize
    bubbles: list[list[float]]
    bubble_confs: list[float]
    texts: list[list[float]]
    text_confs: list[float]
