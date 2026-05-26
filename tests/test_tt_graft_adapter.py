import torch
from types import SimpleNamespace

from scripts.benchmark_drm_g_phase16_graftblock import _metadata
from saint.adapters.drm_grafting_tt_adapter import TTGraftBlock, TTLinear, make_tt_graft_blocks
from saint.adapters.drm_grafting_graftblock_routed import _new_grafts


def test_tt_linear_preserves_batch_shape_and_uses_fewer_parameters_than_dense():
    layer = TTLinear(torch, width=64, input_dims=(8, 8), output_dims=(8, 8), bond_dim=4, seed=11)
    x = torch.randn(2, 3, 64)

    y = layer(x)

    assert y.shape == x.shape
    assert layer.parameter_count() < 64 * 64


def test_tt_graft_block_is_residual_and_recomposable_from_state_dict():
    graft = TTGraftBlock(torch, d_model=16, adapter_width=8, bond_dim=2, seed=5, init_scale=0.1)
    x = torch.randn(2, 4, 16)
    original = graft.hook(None, None, x)
    with torch.no_grad():
        graft.project_up.fill_(0.01)
    changed = graft.hook(None, None, x)

    clone = TTGraftBlock(torch, d_model=16, adapter_width=8, bond_dim=2, seed=99, init_scale=1.0)
    clone.load_state_dict(graft.state_dict(), "cpu")
    recomposed = clone.hook(None, None, x)

    assert torch.allclose(original, x)
    assert not torch.allclose(changed, x)
    assert torch.allclose(recomposed, changed)
    assert clone.adapter_width == 8
    assert clone.bond_dim == 2


def test_make_tt_graft_blocks_returns_seeded_trainable_blocks():
    grafts = make_tt_graft_blocks(
        torch,
        d_model=16,
        adapter_width=8,
        bond_dim=2,
        graft_count=3,
        seed=7,
        init_scale=0.01,
        activation="silu",
        device="cpu",
    )

    assert len(grafts) == 3
    assert all(graft.parameter_count() > 0 for graft in grafts)
    assert all(param.requires_grad for graft in grafts for param in graft.parameters())


def test_routed_graft_factory_can_create_tt_mps_adapters():
    drm_config = SimpleNamespace(d_model=16)
    metadata = {"seed": 17, "device": "cpu"}
    args = SimpleNamespace(
        adapter_type="tt_mps",
        hidden_size=32,
        tt_adapter_width=8,
        tt_bond_dim=2,
        graft_count=2,
        init_scale=0.01,
        activation="silu",
    )

    grafts = _new_grafts(torch, drm_config, metadata, args)

    assert [type(graft).__name__ for graft in grafts] == ["TTGraftBlock", "TTGraftBlock"]
    assert all(graft.adapter_width == 8 for graft in grafts)
