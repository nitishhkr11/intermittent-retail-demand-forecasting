from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.expanded_prophet import block_conformal_interval_offset, conformal_interval_offset


def test_pointwise_conformal_offset_uses_finite_sample_rank():
    assert conformal_interval_offset([1, 5, 8], [0, 0, 10], [2, 4, 9], 0.5) == 1.0


def test_block_conformal_offset_uses_one_max_score_per_horizon():
    blocks = [
        ([1, 5, 8], [0, 4, 7], [2, 6, 10]),
        ([4, 6, 5], [3, 5, 4], [5, 7, 6]),
        ([0, 4, 12], [-1, 3, 10], [1, 5, 11]),
    ]
    # Three block maxima are 0, 0, and 1; at 80% the finite-sample rank is capped to 3.
    assert block_conformal_interval_offset(blocks, 0.8) == 1.0


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1])
def test_block_conformal_offset_rejects_invalid_coverage(coverage):
    with pytest.raises(ValueError, match="between zero and one"):
        block_conformal_interval_offset([([1], [0], [2])], coverage)


def test_block_conformal_offset_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        block_conformal_interval_offset([([np.nan], [0], [2])])
