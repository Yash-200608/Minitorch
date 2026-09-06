import math

import numpy as np
import pytest

from minitorch.rotation import (
    HadamardTransform,
    PermutationTransform,
    RotationTransform,
    SignFlipTransform,
    build_rotation_transform,
    walsh_hadamard_1d,
)


def test_permutation_inverse():
    permutation = PermutationTransform(np.array([2, 0, 3, 1]))
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(permutation.inverse(permutation.apply(x)), x)


def test_sign_flip_inverse_and_abs_preserved():
    signs = SignFlipTransform(np.array([1.0, -1.0, 1.0, -1.0]))
    x = np.array([1.0, 2.0, 3.0, 4.0])
    flipped = signs.apply(x)
    np.testing.assert_allclose(signs.inverse(flipped), x)
    np.testing.assert_allclose(np.abs(flipped), np.abs(x))


def test_hadamard_known_vectors():
    sqrt2 = math.sqrt(2.0)
    np.testing.assert_allclose(walsh_hadamard_1d(np.array([1.0, 1.0])), [sqrt2, 0.0])
    np.testing.assert_allclose(
        walsh_hadamard_1d(np.array([1.0, -1.0])),
        [0.0, sqrt2],
    )


def test_hadamard_is_self_inverse():
    x = np.random.default_rng(0).normal(size=16)
    hadamard = HadamardTransform()
    np.testing.assert_allclose(hadamard.inverse(hadamard.apply(x)), x, atol=1e-10)


def test_rotation_round_trip():
    transform = build_rotation_transform(16, kind="combined")
    x = np.random.default_rng(1).normal(size=(8, 16))
    np.testing.assert_allclose(transform.inverse(transform.apply(x)), x, atol=1e-10)


def test_rotation_preserves_l2_norm():
    transform = build_rotation_transform(16, kind="combined")
    x = np.random.default_rng(2).normal(size=(4, 16))
    original_norm = np.linalg.norm(x)
    rotated_norm = np.linalg.norm(transform.apply(x))
    assert abs(original_norm - rotated_norm) < 1e-10


def test_non_power_of_two_rejected():
    with pytest.raises(ValueError, match="power-of-two"):
        build_rotation_transform(10)


def test_batch_axis_independence():
    transform = build_rotation_transform(4, kind="combined")
    x = np.arange(32.0).reshape(8, 4)
    rotated = transform.apply(x)
    for index in range(8):
        row = x[index]
        np.testing.assert_allclose(
            transform.inverse(rotated[index]),
            row,
            atol=1e-10,
        )
