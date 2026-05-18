"""Shared linear training operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from saint.blocks.metrics import reconstruction_error
from saint.blocks.partition import Matrix
from saint.reconstruction.matrix_ops import add, matvec, shape, zeros
from saint.training.data import LinearTask


@dataclass(frozen=True)
class TrainingResult:
    name: str
    train_loss: float
    test_loss: float
    weight_relative_l1_error: float
    parameter_count: int
    optimizer_state_values: int
    elapsed_s: float
    metadata: dict = field(default_factory=dict)


def mse_loss(weight: Matrix, inputs: list[list[float]], targets: list[list[float]]) -> float:
    rows, _cols = shape(weight)
    total = 0.0
    for vector, target in zip(inputs, targets):
        prediction = matvec(weight, vector)
        for row in range(rows):
            error = prediction[row] - target[row]
            total += error * error
    return total / (len(inputs) * rows)


def frozen_base_test_loss(task: LinearTask) -> float:
    """Loss when no delta is applied to the frozen base weight."""

    return mse_loss(task.base_weight, task.test_inputs, task.test_targets)


def loss_and_delta_gradient(
    task: LinearTask,
    delta: Matrix,
) -> tuple[float, list[list[float]]]:
    weight = add(task.base_weight, delta)
    rows, cols = shape(weight)
    gradient = zeros(rows, cols)
    total = 0.0
    scale = 2.0 / (len(task.train_inputs) * rows)

    for vector, target in zip(task.train_inputs, task.train_targets):
        prediction = matvec(weight, vector)
        for row in range(rows):
            error = prediction[row] - target[row]
            total += error * error
            for col in range(cols):
                gradient[row][col] += scale * error * vector[col]

    return total / (len(task.train_inputs) * rows), gradient


def weight_error(task: LinearTask, delta: Matrix) -> float:
    learned_weight = add(task.base_weight, delta)
    return reconstruction_error(task.target_weight, learned_weight).relative_l1_error


def training_result(
    name: str,
    task: LinearTask,
    delta: Matrix,
    parameter_count: int,
    start: float,
    metadata: dict | None = None,
    baseline_test_loss: float | None = None,
) -> TrainingResult:
    weight = add(task.base_weight, delta)
    test_loss = mse_loss(weight, task.test_inputs, task.test_targets)
    baseline = frozen_base_test_loss(task) if baseline_test_loss is None else baseline_test_loss
    gain = max(baseline - test_loss, 0.0)
    gain_per_parameter = gain / parameter_count if parameter_count > 0 else 0.0
    full_metadata = dict(metadata or {})
    full_metadata.setdefault("baseline_test_loss", baseline)
    full_metadata.setdefault("test_loss_gain", gain)
    full_metadata.setdefault("gain_per_parameter", gain_per_parameter)
    return TrainingResult(
        name=name,
        train_loss=mse_loss(weight, task.train_inputs, task.train_targets),
        test_loss=test_loss,
        weight_relative_l1_error=weight_error(task, delta),
        parameter_count=parameter_count,
        optimizer_state_values=parameter_count * 2,
        elapsed_s=perf_counter() - start,
        metadata=full_metadata,
    )


def top_gradient_positions(
    gradient: Matrix,
    budget: int,
) -> set[tuple[int, int]]:
    rows, cols = shape(gradient)
    ranked = sorted(
        (
            (abs(gradient[row][col]), row, col)
            for row in range(rows)
            for col in range(cols)
        ),
        reverse=True,
    )
    return {(row, col) for _score, row, col in ranked[:budget]}
