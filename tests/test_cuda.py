"""These test cases are meant to check CPU/GPU consistency and basic
functionality, not to validate the models' accuracy.

To update snapshots, run `pytest --snapshot-update`.

Small differences in floating-point precision may occur between CPU and GPU
computations. We limit these difference by comparing with a relative tolerance
of 1e-3 and absolute tolerance of 1e-2.

Test images are royalty free images from:

https://pixabay.com/photos/mass-people-supporters-1355493/
https://pixabay.com/photos/train-transit-railway-rail-journey-5115906/

"""

import json
from pathlib import Path

import numpy as np
import pytest

from lwcc import LWCC

HERE = Path(__file__).parent


VALID_MODELS_WEIGHTS = [
    ("CSRNet", "SHA"),
    ("CSRNet", "SHB"),
    ("Bay", "SHA"),
    ("Bay", "SHB"),
    ("Bay", "QNRF"),
    ("DM-Count", "SHA"),
    ("DM-Count", "SHB"),
    ("DM-Count", "QNRF"),
    ("SFANet", "SHB"),
]


@pytest.mark.parametrize("model_name,model_weights", VALID_MODELS_WEIGHTS)
def test_lwcc_sparse(model_name, model_weights, snapshot):
    img = HERE / "samples" / "sparse.jpg"
    cpu_result, cpu_density = LWCC.get_count(
        device="cpu",
        img_paths=img,
        model_weights=model_weights,
        model_name=model_name,
        return_density=True,
    )

    cuda_result, cuda_density = LWCC.get_count(
        device="cuda",
        img_paths=img,
        model_weights=model_weights,
        model_name=model_name,
        return_density=True,
    )

    assert cpu_density is not None
    assert cuda_density is not None

    assert pytest.approx(cpu_result, rel=1e-3, abs=1e-2) == cuda_result

    snapshot.assert_match(json.dumps(cpu_result), "counts.json")


@pytest.mark.parametrize("model_name,model_weights", VALID_MODELS_WEIGHTS)
def test_lwcc_dense(model_name, model_weights, snapshot):
    img = HERE / "samples" / "dense.jpg"
    cpu_result, cpu_density = LWCC.get_count(
        device="cpu",
        img_paths=img,
        model_weights=model_weights,
        model_name=model_name,
        return_density=True,
        resize_img=False,
    )

    cuda_result, cuda_density = LWCC.get_count(
        device="cuda",
        img_paths=img,
        model_weights=model_weights,
        model_name=model_name,
        return_density=True,
        resize_img=False,
    )

    assert cpu_density is not None
    assert cuda_density is not None

    assert pytest.approx(cpu_result, rel=1e-3, abs=1e-2) == cuda_result

    snapshot.assert_match(json.dumps(cpu_result), "counts.json")
