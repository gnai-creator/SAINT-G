"""Tensor-Train / MPS-style graft adapters for Phase 16 Marco 4O."""

from __future__ import annotations

from typing import Any, Iterable


def _factor_pair(width: int) -> tuple[int, int]:
    """Return a compact two-factor tensorization for ``width``."""
    width = int(width)
    if width < 1:
        raise ValueError("width must be >= 1")
    root = int(width**0.5)
    for first in range(root, 0, -1):
        if width % first == 0:
            return first, width // first
    return 1, width


def _tuple_product(values: Iterable[int]) -> int:
    product = 1
    for value in values:
        product *= int(value)
    return int(product)


class TTLinear:
    """Small Tensor-Train linear map that materializes its dense matrix on demand.

    The implementation intentionally targets the small adapter widths used by
    Marco 4O (64/128/256), keeping the code simple and auditable while still
    enforcing a bounded TT/MPS parameterization through the trainable cores.
    """

    def __init__(
        self,
        torch,
        *,
        width: int,
        input_dims: tuple[int, ...] | None = None,
        output_dims: tuple[int, ...] | None = None,
        bond_dim: int = 4,
        seed: int = 0,
        init_scale: float = 0.02,
    ):
        self.torch = torch
        self.width = int(width)
        self.input_dims = tuple(int(v) for v in (input_dims or _factor_pair(self.width)))
        self.output_dims = tuple(int(v) for v in (output_dims or self.input_dims))
        self.bond_dim = int(bond_dim)
        if self.width < 1:
            raise ValueError("width must be >= 1")
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be >= 1")
        if _tuple_product(self.input_dims) != self.width:
            raise ValueError("input_dims product must equal width")
        if _tuple_product(self.output_dims) != self.width:
            raise ValueError("output_dims product must equal width")
        if len(self.input_dims) != len(self.output_dims):
            raise ValueError("input_dims and output_dims must have the same rank")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        ranks = [1] + [self.bond_dim] * (len(self.input_dims) - 1) + [1]
        self.cores = []
        for index, (in_dim, out_dim) in enumerate(zip(self.input_dims, self.output_dims)):
            core = torch.randn(
                ranks[index],
                int(in_dim),
                int(out_dim),
                ranks[index + 1],
                generator=generator,
            ) * float(init_scale)
            self.cores.append(torch.nn.Parameter(core))

    def parameters(self):
        return tuple(self.cores)

    def to(self, device: str):
        for core in self.cores:
            core.data = core.data.to(device)
        return self

    def parameter_count(self) -> int:
        return int(sum(core.numel() for core in self.cores))

    def dense_weight(self):
        matrix = self.cores[0]
        for core in self.cores[1:]:
            matrix = self.torch.tensordot(matrix, core, dims=([-1], [0]))
        matrix = matrix.squeeze(0).squeeze(-1)
        rank = len(self.input_dims)
        perm = []
        for index in range(rank):
            perm.extend([index * 2, index * 2 + 1])
        matrix = matrix.permute(*perm).contiguous()
        return matrix.reshape(self.width, self.width)

    def __call__(self, value):
        return value.matmul(self.dense_weight())

    def state_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "bond_dim": self.bond_dim,
            "cores": [core.detach().cpu() for core in self.cores],
        }

    def load_state_dict(self, state: dict[str, Any], device: str):
        self.width = int(state.get("width", self.width))
        self.input_dims = tuple(int(v) for v in state.get("input_dims", self.input_dims))
        self.output_dims = tuple(int(v) for v in state.get("output_dims", self.output_dims))
        self.bond_dim = int(state.get("bond_dim", self.bond_dim))
        for core, incoming in zip(self.cores, state["cores"]):
            core.data = incoming.to(device)
        return self


class TTGraftBlock:
    """Residual adapter: output + scale * project_up(TT(act(project_down(output))))."""

    def __init__(
        self,
        torch,
        *,
        d_model: int,
        adapter_width: int,
        bond_dim: int,
        seed: int,
        init_scale: float = 0.01,
        activation: str = "silu",
        input_dims: tuple[int, ...] | None = None,
    ):
        self.torch = torch
        self.d_model = int(d_model)
        self.adapter_width = int(adapter_width)
        self.bond_dim = int(bond_dim)
        self.activation = str(activation)
        self.input_dims = tuple(int(v) for v in (input_dims or _factor_pair(self.adapter_width)))
        self.enabled = True
        self.runtime_scale = 1.0
        if self.d_model < 1 or self.adapter_width < 1:
            raise ValueError("d_model and adapter_width must be >= 1")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.project_down = torch.nn.Parameter(
            torch.randn(self.d_model, self.adapter_width, generator=generator) / max(1, self.d_model)
        )
        self.tt = TTLinear(
            torch,
            width=self.adapter_width,
            input_dims=self.input_dims,
            output_dims=self.input_dims,
            bond_dim=self.bond_dim,
            seed=int(seed) + 1009,
            init_scale=init_scale,
        )
        self.project_up = torch.nn.Parameter(torch.zeros(self.adapter_width, self.d_model))
        self.scale = torch.nn.Parameter(torch.tensor(float(init_scale)))

    def parameters(self):
        return (self.project_down, *self.tt.parameters(), self.project_up, self.scale)

    def to(self, device: str):
        self.project_down.data = self.project_down.data.to(device)
        self.project_up.data = self.project_up.data.to(device)
        self.scale.data = self.scale.data.to(device)
        self.tt.to(device)
        return self

    def parameter_count(self) -> int:
        return int(
            self.project_down.numel()
            + self.project_up.numel()
            + self.scale.numel()
            + self.tt.parameter_count()
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "adapter_type": "tt_mps",
            "project_down": self.project_down.detach().cpu(),
            "project_up": self.project_up.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "tt": self.tt.state_dict(),
            "d_model": self.d_model,
            "adapter_width": self.adapter_width,
            "bond_dim": self.bond_dim,
            "activation": self.activation,
            "input_dims": self.input_dims,
        }

    def load_state_dict(self, state: dict[str, Any], device: str):
        self.project_down.data = state["project_down"].to(device)
        self.project_up.data = state["project_up"].to(device)
        self.scale.data = state["scale"].to(device)
        self.tt.load_state_dict(state["tt"], device)
        self.d_model = int(state.get("d_model", self.project_down.shape[0]))
        self.adapter_width = int(state.get("adapter_width", self.project_down.shape[1]))
        self.bond_dim = int(state.get("bond_dim", self.bond_dim))
        self.activation = str(state.get("activation", self.activation))
        self.input_dims = tuple(int(v) for v in state.get("input_dims", self.input_dims))
        return self

    def _activate(self, value):
        if self.activation == "gelu":
            return self.torch.nn.functional.gelu(value)
        if self.activation == "relu":
            return self.torch.relu(value)
        return self.torch.nn.functional.silu(value)

    def hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if not self.enabled:
            return output
        hidden = self._activate(output.matmul(self.project_down))
        transformed = self.tt(hidden)
        delta = transformed.matmul(self.project_up)
        return output + float(self.runtime_scale) * self.scale * delta


def make_tt_graft_blocks(
    torch,
    *,
    d_model: int,
    adapter_width: int,
    bond_dim: int,
    graft_count: int,
    seed: int,
    init_scale: float,
    activation: str,
    device: str,
) -> list[TTGraftBlock]:
    return [
        TTGraftBlock(
            torch,
            d_model=d_model,
            adapter_width=adapter_width,
            bond_dim=bond_dim,
            seed=int(seed) + index,
            init_scale=init_scale,
            activation=activation,
        ).to(device)
        for index in range(int(graft_count))
    ]
