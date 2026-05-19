"""Full-budget linear baselines for DRM-G grafting."""

from __future__ import annotations


class FullBudgetLinearGraft:
    def __init__(self, torch, target, budget: int):
        self.torch = torch
        self.out_features = int(target.weight.shape[0])
        self.in_features = int(target.weight.shape[1])
        self.bias_enabled = getattr(target, "bias", None) is not None
        count = max(1, min(int(budget), self.out_features * self.in_features))
        rows = []
        cols = []
        for index in range(count):
            rows.append(index // self.in_features)
            cols.append(index % self.in_features)
        self.rows = torch.tensor(rows, dtype=torch.long)
        self.cols = torch.tensor(cols, dtype=torch.long)
        self.values = torch.nn.Parameter(torch.zeros(count))

    def to(self, device: str):
        self.rows = self.rows.to(device)
        self.cols = self.cols.to(device)
        self.values = self.torch.nn.Parameter(self.values.detach().to(device))
        return self

    def parameters(self):
        return [self.values]

    def hook(self, _module, inputs, output):
        delta = self.torch.zeros(
            self.out_features,
            self.in_features,
            device=output.device,
            dtype=output.dtype,
        )
        delta[self.rows, self.cols] = self.values.to(output.dtype)
        update = self.torch.nn.functional.linear(inputs[0], delta)
        return output + update


__all__ = ["FullBudgetLinearGraft"]
