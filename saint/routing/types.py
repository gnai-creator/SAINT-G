"""Routing dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from saint.reconstruction.baselines import ReconstructionResult


@dataclass(frozen=True)
class RoutedRegion:
    row_start: int
    col_start: int
    height: int
    width: int
    method: str
    relative_l1_error: float
    parameter_count: int


@dataclass(frozen=True)
class RoutingPlan:
    regions: tuple[RoutedRegion, ...]
    total_parameter_count: int
    metadata: dict


@dataclass(frozen=True)
class WeightSearchResult:
    parameter_weight: float
    result: ReconstructionResult
    relative_l1_error: float
    compression_ratio: float
    target_met: bool


@dataclass(frozen=True)
class RegionCandidate:
    method: str
    reconstructed: list[list[float]]
    relative_l1_error: float
    parameter_count: int
    score: float
