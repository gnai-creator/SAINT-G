"""Compatibility exports for SAINT block routing."""

from saint.routing.budget import route_matrix_regions_by_budget
from saint.routing.quality import route_matrix_regions
from saint.routing.reconstruction import (
    routed_budget_reconstruction,
    routed_codebook_reconstruction,
    routed_sensitivity_budget_reconstruction,
    search_routed_budget_reconstruction,
)
from saint.routing.sensitivity import route_matrix_regions_by_sensitivity_budget
from saint.routing.types import (
    RegionCandidate,
    RoutedRegion,
    RoutingPlan,
    WeightSearchResult,
)

__all__ = [
    "RegionCandidate",
    "RoutedRegion",
    "RoutingPlan",
    "WeightSearchResult",
    "route_matrix_regions",
    "route_matrix_regions_by_budget",
    "route_matrix_regions_by_sensitivity_budget",
    "routed_budget_reconstruction",
    "routed_codebook_reconstruction",
    "routed_sensitivity_budget_reconstruction",
    "search_routed_budget_reconstruction",
]
