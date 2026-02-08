"""Erase 서비스 단위 테스트"""

import numpy as np
import pytest

from src.services.erase import EraseError, ensure_grayscale_mask


class TestEnsureGrayscaleMask:
    """ensure_grayscale_mask 분기 커버리지 테스트"""

    def test_2d_mask_passthrough(self) -> None:
        """(H, W) 형태는 그대로 반환"""
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = ensure_grayscale_mask(mask)

        assert result.shape == (100, 100)
        assert np.array_equal(result, mask)

    def test_single_channel_mask(self) -> None:
        """(H, W, 1) 형태는 (H, W)로 변환"""
        mask = np.full((100, 100, 1), 255, dtype=np.uint8)
        result = ensure_grayscale_mask(mask)

        assert result.shape == (100, 100)
        assert result[0, 0] == 255

    def test_rgb_mask(self) -> None:
        """(H, W, 3) RGB는 grayscale로 변환"""
        mask = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = ensure_grayscale_mask(mask)

        assert result.shape == (100, 100)

    def test_rgba_mask(self) -> None:
        """(H, W, 4) RGBA는 grayscale로 변환"""
        mask = np.full((100, 100, 4), 255, dtype=np.uint8)
        result = ensure_grayscale_mask(mask)

        assert result.shape == (100, 100)

    def test_invalid_channel_count(self) -> None:
        """지원하지 않는 채널 수는 에러"""
        mask = np.full((100, 100, 5), 255, dtype=np.uint8)

        with pytest.raises(EraseError) as exc_info:
            ensure_grayscale_mask(mask)

        assert exc_info.value.code == "INPAINTING_FAILED"
