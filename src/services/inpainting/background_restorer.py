"""IOPaint 기반 배경 복원"""

import base64
import io

import cv2
import httpx
import numpy as np
from PIL import Image

from src.schemas.pipeline import TextRegion
from src.services.inpainting.solid_fill import InpaintingError
from src.services.inpainting.utils import (
    calc_render_bbox,
    clip_to_bounds,
    convert_to_bgr,
    create_mask,
)


class IOPaintRestorer:
    """IOPaint HuggingFace Space를 사용한 배경 복원"""

    def __init__(
        self,
        space_url: str,
        timeout: int = 120,
    ):
        self._space_url = space_url.rstrip("/")
        self._timeout = timeout

    def restore(
        self, image: np.ndarray, regions: list[TextRegion]
    ) -> tuple[np.ndarray, list[TextRegion]]:
        if not regions:
            return image, []

        h, w = image.shape[:2]
        updated: list[TextRegion] = []

        for region in regions:
            inpaint_bbox = clip_to_bounds(region.text_bbox, w, h)
            if inpaint_bbox.width <= 0 or inpaint_bbox.height <= 0:
                continue
            render_bbox = calc_render_bbox(region.bubble_bbox, inpaint_bbox)
            updated.append(
                TextRegion(
                    index=region.index,
                    text_bbox=region.text_bbox,
                    bubble_bbox=region.bubble_bbox,
                    inpaint_bbox=inpaint_bbox,
                    render_bbox=render_bbox,
                )
            )

        if not updated:
            return image, []

        mask = create_mask((h, w), updated)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        clean = self._call_api(image_rgb, mask)

        return clean, updated

    def restore_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result_bgr = self._call_api(image, mask)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    def _call_api(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img_b64 = self._to_base64(image_rgb)
        mask_b64 = self._to_base64(mask)

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._space_url}/api/v1/inpaint",
                    json={"image": img_b64, "mask": mask_b64},
                )
                resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise InpaintingError("IOPaint API 타임아웃") from e
        except httpx.HTTPStatusError as e:
            raise InpaintingError(f"IOPaint API 오류: {e.response.status_code}") from e
        except Exception as e:
            raise InpaintingError(f"IOPaint API 호출 실패: {e}") from e

        return self._parse_response(resp.content)

    def _to_base64(self, arr: np.ndarray) -> str:
        img_pil = Image.fromarray(arr)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _parse_response(self, content: bytes) -> np.ndarray:
        try:
            img_pil = Image.open(io.BytesIO(content))
            img_rgb = np.array(img_pil)
            return convert_to_bgr(img_rgb)
        except Exception as e:
            raise InpaintingError(f"결과 이미지 변환 실패: {e}") from e
