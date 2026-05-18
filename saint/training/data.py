"""Linear task data generation for phase-4 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from saint.blocks.partition import Matrix
from saint.reconstruction.matrix_ops import add, matvec, zeros


@dataclass(frozen=True)
class LinearTask:
    base_weight: list[list[float]]
    target_weight: list[list[float]]
    train_inputs: list[list[float]]
    train_targets: list[list[float]]
    test_inputs: list[list[float]]
    test_targets: list[list[float]]


def _random_matrix(rows: int, cols: int, rng: Random, scale: float) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _repeated_delta(rows: int, cols: int, rng: Random) -> list[list[float]]:
    prototypes = (
        ((0.08, -0.04), (0.02, 0.06)),
        ((-0.05, 0.03), (0.04, -0.02)),
        ((0.00, 0.00), (0.00, 0.00)),
    )
    delta = zeros(rows, cols)
    for row in range(0, rows, 2):
        for col in range(0, cols, 2):
            prototype = prototypes[rng.randrange(len(prototypes))]
            factor = rng.choice((0.5, 1.0, 1.5))
            for r_offset in range(min(2, rows - row)):
                for c_offset in range(min(2, cols - col)):
                    delta[row + r_offset][col + c_offset] = (
                        prototype[r_offset][c_offset] * factor
                    )
    return delta


def _dense_delta(rows: int, cols: int, rng: Random) -> list[list[float]]:
    return _random_matrix(rows, cols, rng, scale=0.07)


def _make_inputs(count: int, cols: int, rng: Random) -> list[list[float]]:
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(count)]


def _targets(weight: Matrix, inputs: list[list[float]]) -> list[list[float]]:
    return [matvec(weight, vector) for vector in inputs]


def make_linear_delta_task(
    *,
    rows: int = 8,
    cols: int = 8,
    train_samples: int = 96,
    test_samples: int = 32,
    seed: int = 11,
    delta_mode: str = "repeated",
) -> LinearTask:
    """Create a deterministic teacher task y = W_target x."""

    rng = Random(seed)
    base_weight = _random_matrix(rows, cols, rng, scale=0.12)
    if delta_mode == "repeated":
        delta = _repeated_delta(rows, cols, rng)
    elif delta_mode == "dense":
        delta = _dense_delta(rows, cols, rng)
    else:
        raise ValueError(f"unknown delta_mode: {delta_mode}")
    target_weight = add(base_weight, delta)
    train_inputs = _make_inputs(train_samples, cols, rng)
    test_inputs = _make_inputs(test_samples, cols, rng)
    return LinearTask(
        base_weight=base_weight,
        target_weight=target_weight,
        train_inputs=train_inputs,
        train_targets=_targets(target_weight, train_inputs),
        test_inputs=test_inputs,
        test_targets=_targets(target_weight, test_inputs),
    )
