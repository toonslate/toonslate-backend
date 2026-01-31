"""IOPaint LaMa Inpainting (HuggingFace Space)"""

import base64
import io
from pathlib import Path

import cv2
import httpx
import numpy as np
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

IOPAINT_SPACE_URL = "https://sanster-iopaint-lama.hf.space"


class IOPaintLamaInpainting:
    """IOPaint HuggingFace Space를 사용한 LaMa 인페인팅"""

    def __init__(
        self,
        space_url: str = IOPAINT_SPACE_URL,
        timeout: int = 120,
        debug_dir: str | None = None,
    ):
        self.space_url = space_url.rstrip("/")
        self.timeout = timeout
        self.debug_dir = Path(debug_dir) if debug_dir else None

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

        clean_image = self._call_iopaint(image_rgb, mask)

        return clean_image, updated_regions

    def _calc_inpaint_bbox(self, text: BBox, img_size: tuple[int, int]) -> BBox:
        """text_bbox 기반 inpaint 영역 계산 (padding 없음)"""
        w, h = img_size
        return clip_to_bounds(text, w, h)

    def _call_iopaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """IOPaint API 호출 및 결과 이미지 반환"""
        img_b64 = self._to_base64(image_rgb)
        mask_b64 = self._to_base64(mask)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.space_url}/api/v1/inpaint",
                    json={"image": img_b64, "mask": mask_b64},
                )
                resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise InpaintingError("IOPaint API 타임아웃 (Space가 sleep 상태일 수 있음)") from e
        except httpx.HTTPStatusError as e:
            raise InpaintingError(f"IOPaint API 오류: {e.response.status_code}") from e
        except Exception as e:
            raise InpaintingError(f"IOPaint API 호출 실패: {e}") from e

        return self._parse_response(resp.content)

    def _to_base64(self, arr: np.ndarray) -> str:
        """numpy 배열을 base64 PNG 문자열로 변환"""
        img_pil = Image.fromarray(arr)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _parse_response(self, content: bytes) -> np.ndarray:
        """PNG 바이너리 응답을 BGR numpy 배열로 변환"""
        try:
            img_pil = Image.open(io.BytesIO(content))
            img_rgb = np.array(img_pil)
            return convert_to_bgr(img_rgb)
        except Exception as e:
            raise InpaintingError(f"결과 이미지 변환 실패: {e}") from e
