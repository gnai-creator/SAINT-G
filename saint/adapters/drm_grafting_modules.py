"""Graft modules used by DRM-G experiments."""

from __future__ import annotations

from typing import Any


class PhiHiddenGraft:
    def __init__(
        self,
        torch,
        d_model: int,
        rank: int,
        scale: float,
        seed: int,
        projections: tuple[Any, Any] | None = None,
    ):
        self.torch = torch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        if projections is None:
            self.left = torch.randn(d_model, rank, generator=generator) / max(1, d_model)
            self.right = torch.randn(rank, d_model, generator=generator) / max(1, rank)
        else:
            self.left, self.right = projections
        self.phi = torch.nn.Parameter(torch.zeros(rank, rank))
        self.scale = scale

    def to(self, device: str):
        self.left = self.left.to(device)
        self.right = self.right.to(device)
        self.phi = self.torch.nn.Parameter(self.phi.detach().to(device))
        return self

    def parameters(self):
        return [self.phi]

    def hook(self, _module, _inputs, output):
        delta = output.matmul(self.left).matmul(self.phi).matmul(self.right)
        return output + self.scale * delta

    def payload(self, target_module: str, projection_mode: str) -> dict[str, Any]:
        return {
            "format": "drm_graft_payload",
            "target_module": target_module,
            "projection_init": projection_mode,
            "scale": self.scale,
            "trainable_parameters": int(self.phi.numel()),
            "left": self.left.detach().cpu().tolist(),
            "phi": self.phi.detach().cpu().tolist(),
            "right": self.right.detach().cpu().tolist(),
        }

    @classmethod
    def from_payload(cls, torch, payload: dict[str, Any]):
        left = torch.tensor(payload["left"], dtype=torch.float32)
        phi = torch.tensor(payload["phi"], dtype=torch.float32)
        right = torch.tensor(payload["right"], dtype=torch.float32)
        graft = cls(
            torch,
            d_model=int(left.shape[0]),
            rank=int(phi.shape[0]),
            scale=float(payload.get("scale", 1.0)),
            seed=0,
            projections=(left, right),
        )
        graft.phi = torch.nn.Parameter(phi)
        return graft


class DenseBudgetGraft:
    def __init__(self, torch, d_model: int, rank: int, scale: float):
        self.weight = torch.nn.Parameter(torch.zeros(rank, rank))
        self.d_model = d_model
        self.rank = rank
        self.scale = scale
        self.torch = torch

    def to(self, device: str):
        self.weight = self.torch.nn.Parameter(self.weight.detach().to(device))
        return self

    def parameters(self):
        return [self.weight]

    def hook(self, _module, _inputs, output):
        update = self.torch.zeros(self.d_model, self.d_model, device=output.device)
        update[: self.rank, : self.rank] = self.weight
        return output + self.scale * output.matmul(update)


__all__ = ["DenseBudgetGraft", "PhiHiddenGraft"]
