"""멀티 모델 번역 파이프라인 테스트"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.schemas.pipeline import BBox, TextRegion, TranslationResult
from src.services.detection import set_detection
from src.services.detection.schemas import DetectionResult, ImageSize
from src.services.inpainting import set_inpainting
from src.services.pipeline import PipelineError, build_text_regions, translate_image
from src.services.translation import set_translation


def _detection(
    texts: list[list[float]] | None = None,
    bubbles: list[list[float]] | None = None,
) -> DetectionResult:
    texts = texts or []
    bubbles = bubbles or []
    return DetectionResult(
        image_size=ImageSize(width=100, height=100),
        texts=texts,
        text_confs=[0.9] * len(texts),
        bubbles=bubbles,
        bubble_confs=[0.9] * len(bubbles),
    )


class TestBuildTextRegions:
    def test_basic_conversion(self) -> None:
        detection = _detection(
            texts=[[10, 10, 50, 50], [60, 60, 90, 90]],
            bubbles=[[0, 0, 100, 100]],
        )
        regions, bubbles = build_text_regions(detection)

        assert len(regions) == 2
        assert regions[0].index == 0
        assert regions[0].text_bbox == BBox(x1=10, y1=10, x2=50, y2=50)
        assert regions[1].index == 1
        assert len(bubbles) == 1
        assert bubbles[0] == BBox(x1=0, y1=0, x2=100, y2=100)

    def test_empty_texts(self) -> None:
        detection = _detection(bubbles=[[0, 0, 100, 100]])
        regions, bubbles = build_text_regions(detection)

        assert regions == []
        assert len(bubbles) == 1

    def test_empty_both(self) -> None:
        detection = _detection()
        regions, bubbles = build_text_regions(detection)

        assert regions == []
        assert bubbles == []

    def test_index_ordering(self) -> None:
        detection = _detection(texts=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])
        regions, _ = build_text_regions(detection)

        assert [r.index for r in regions] == [0, 1, 2]


class FakeDetector:
    def __init__(self, detection: DetectionResult) -> None:
        self._detection = detection

    def detect(self, image_path: str) -> DetectionResult:
        return self._detection


class FakeInpainter:
    def inpaint(
        self,
        image: np.ndarray,
        text_regions: list[TextRegion],
        bubble_bboxes: list[BBox],
    ) -> tuple[np.ndarray, list[TextRegion]]:
        updated = [
            TextRegion(
                index=r.index,
                text_bbox=r.text_bbox,
                render_bbox=r.text_bbox,
            )
            for r in text_regions
        ]
        return image, updated

    def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UnreachableInpainter:
    def inpaint(
        self,
        image: np.ndarray,
        text_regions: list[TextRegion],
        bubble_bboxes: list[BBox],
    ) -> tuple[np.ndarray, list[TextRegion]]:
        raise AssertionError("Inpainting이 호출되면 안 됨")

    def inpaint_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise AssertionError("Inpainting이 호출되면 안 됨")


class UnreachableTranslator:
    def translate(self, image_path: str, bboxes: list[BBox]) -> list[TranslationResult]:
        raise AssertionError("Translation이 호출되면 안 됨")


class FakeTranslator:
    def __init__(self, translations: list[TranslationResult]) -> None:
        self._translations = translations

    def translate(self, image_path: str, bboxes: list[BBox]) -> list[TranslationResult]:
        return self._translations


class TestTranslateImage:
    def setup_method(self) -> None:
        set_detection(None)
        set_translation(None)
        set_inpainting(None)

    def teardown_method(self) -> None:
        set_detection(None)
        set_translation(None)
        set_inpainting(None)

    def test_happy_path(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (100, 100), "white")
        path = str(tmp_path / "test.png")
        img.save(path)

        detection = _detection(
            texts=[[10, 10, 50, 50]],
            bubbles=[[0, 0, 60, 60]],
        )
        set_detection(FakeDetector(detection))
        set_inpainting(FakeInpainter())
        set_translation(FakeTranslator([TranslationResult(index=0, translated="Hello")]))

        result = translate_image(path)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_no_text_returns_original(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (100, 100), "red")
        path = str(tmp_path / "test.png")
        img.save(path)

        set_detection(FakeDetector(_detection()))
        set_inpainting(UnreachableInpainter())
        set_translation(UnreachableTranslator())

        result = translate_image(path)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_image_load_failure_raises_pipeline_error(self, tmp_path: Path) -> None:
        txt_path = str(tmp_path / "not_an_image.txt")
        with open(txt_path, "w") as f:
            f.write("not an image")

        detection = _detection(texts=[[10, 10, 50, 50]])
        set_detection(FakeDetector(detection))

        with pytest.raises(PipelineError, match="이미지를 읽을 수 없음"):
            translate_image(txt_path)
