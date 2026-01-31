"""Replicate LaMa Inpainting"""

import io
import os
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
import replicate
from PIL import Image

from src.schemas.pipeline import BBox, TextRegion
from src.services.inpainting.solid_fill import InpaintingError
from src.services.inpainting.utils import (
    calc_render_bbox,
    clip_to_bounds,
    convert_to_bgr,
    create_mask,
    find_bubble,
    save_debug_images,
)


class _Readable(Protocol):
    """read() 메서드를 가진 객체"""

    def read(self) -> bytes: ...


LAMA_MODEL_ID = "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72"


class ReplicateLamaInpainting:
    """Replicate LaMa API를 사용한 AI 인페인팅"""

    def __init__(self, api_token: str, padding_ratio: float = 0.3, debug_dir: str | None = None):
        self.api_token = api_token
        self.padding_ratio = padding_ratio
        self.debug_dir = Path(debug_dir) if debug_dir else None
        os.environ["REPLICATE_API_TOKEN"] = api_token

        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def inpaint(
        self,
        image: np.ndarray,
        text_regions: list[TextRegion],
        bubble_bboxes: list[BBox],
    ) -> tuple[np.ndarray, list[TextRegion]]:
        if image.size == 0:
            raise InpaintingError("유효하지 않은 이미지입니다")

        h, w = image.shape[:2]

        updated_regions: list[TextRegion] = []

        for region in text_regions:
            bubble = find_bubble(region.text_bbox, bubble_bboxes)
            inpaint_bbox = self._calc_inpaint_bbox(region.text_bbox, (w, h))
            render_bbox = calc_render_bbox(bubble, inpaint_bbox)

            updated_regions.append(
                TextRegion(
                    index=region.index,
                    text_bbox=region.text_bbox,
                    bubble_bbox=bubble,
                    inpaint_bbox=inpaint_bbox,
                    render_bbox=render_bbox,
                )
            )

        mask = create_mask((h, w), updated_regions)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.debug_dir:
            save_debug_images(self.debug_dir, image, mask, updated_regions)

        clean_image = self._call_replicate(image_rgb, mask)

        return clean_image, updated_regions

    def _calc_inpaint_bbox(self, text: BBox, img_size: tuple[int, int]) -> BBox:
        """LaMa용 넉넉한 마스크 영역 - 말풍선 경계 무시"""
        w, h = img_size
        pad_x = text.width * self.padding_ratio
        pad_y = text.height * self.padding_ratio

        bbox = BBox(
            x1=text.x1 - pad_x,
            y1=text.y1 - pad_y,
            x2=text.x2 + pad_x,
            y2=text.y2 + pad_y,
        )
        return clip_to_bounds(bbox, w, h)

    def _call_replicate(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Replicate API 호출 및 결과 이미지 반환"""
        img_buffer = self._to_png_buffer(image_rgb)
        mask_buffer = self._to_png_buffer(mask)

        try:
            output = replicate.run(
                LAMA_MODEL_ID,
                input={"image": img_buffer, "mask": mask_buffer},
            )
        except Exception as e:
            raise InpaintingError(f"Replicate API 호출 실패: {e}") from e

        return self._convert_output(cast(_Readable, output))

    def _to_png_buffer(self, arr: np.ndarray) -> io.BytesIO:
        """numpy 배열을 PNG bytes 버퍼로 변환"""
        img_pil = Image.fromarray(arr)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    def _convert_output(self, output: _Readable) -> np.ndarray:
        """Replicate FileOutput을 BGR numpy 배열로 변환"""
        try:
            img_bytes = output.read()
            img_pil = Image.open(io.BytesIO(img_bytes))
            img_rgb = np.array(img_pil)
            return convert_to_bgr(img_rgb)
        except Exception as e:
            raise InpaintingError(f"결과 이미지 변환 실패: {e}") from e
