import pytest
import torch

try:
    import torch_harmonics  # noqa: F401
except ModuleNotFoundError:
    pytest.skip(
        "Skipping because torch_harmonics is not installed", allow_module_level=True
    )

from ..spherical_convolution import SphericalConv
from ..sph_convolution_modulated import SphericalConvModulated


def _default_embed(dim=8):
    return {"type_t": "sinusoidal", "type_k": "power", "dim": dim,
            "alpha": -2.0, "r": 10000.0}


def _default_mod(mod_type="real", share_m=True, pre_modulate=True):
    return {"enabled": True, "type": mod_type, "hidden_channels": 16,
            "full_res": False, "share_m": share_m,
            "pre_modulate": pre_modulate}


def test_no_modulation_matches_parent():
    """With embed=mode_modulation=None, output equals SphericalConv on shared weights."""
    torch.manual_seed(0)
    n_modes = (6, 6)
    in_channels, out_channels = 2, 3
    spatial = (12, 12)

    base = SphericalConv(
        in_channels, out_channels, n_modes, factorization="dense",
    )
    mod = SphericalConvModulated(
        in_channels, out_channels, n_modes, factorization="dense",
        embed=None, mode_modulation=None,
    )
    mod.load_state_dict(base.state_dict())

    x = torch.randn(2, in_channels, *spatial)
    with torch.no_grad():
        y_base = base(x)
        y_mod_no_t = mod(x)
        y_mod_with_t = mod(x, t=torch.zeros(2, 1))  # t ignored when modulator is None

    torch.testing.assert_close(y_base, y_mod_no_t)
    torch.testing.assert_close(y_base, y_mod_with_t)
    assert mod.modulator is None


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
@pytest.mark.parametrize("share_m", [True, False])
@pytest.mark.parametrize("pre_modulate", [True, False])
def test_modulated_forward_shape(mod_type, share_m, pre_modulate):
    torch.manual_seed(0)
    n_modes = (6, 6)
    in_channels, out_channels = 2, 3
    spatial = (12, 12)

    layer = SphericalConvModulated(
        in_channels, out_channels, n_modes, factorization="dense",
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod(mod_type, share_m=share_m,
                                     pre_modulate=pre_modulate),
    )

    x = torch.randn(2, in_channels, *spatial)
    t = torch.tensor([[0.5], [1.5]])
    y = layer(x, t)

    assert y.shape == (2, out_channels, *spatial)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("type_t,type_k", [
    ("sinusoidal", "power"),
    ("power", "sinusoidal"),
    ("sinusoidal", "sinusoidal"),
    ("power", "power"),
])
def test_modulated_embed_types(type_t, type_k):
    embed = _default_embed(dim=8)
    embed["type_t"] = type_t
    embed["type_k"] = type_k

    layer = SphericalConvModulated(
        2, 3, (6, 6), factorization="dense",
        embed=embed, mode_modulation=_default_mod("real"),
    )

    x = torch.randn(2, 2, 12, 12)
    t = torch.tensor([[0.5], [1.5]])
    y = layer(x, t)
    assert y.shape == (2, 3, 12, 12)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_modulated_backward_grads_all_params(mod_type):
    torch.manual_seed(0)
    layer = SphericalConvModulated(
        2, 3, (6, 6), factorization="dense",
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod(mod_type),
    )
    x = torch.randn(2, 2, 12, 12, requires_grad=True)
    t = torch.tensor([[0.5], [1.5]])

    y = layer(x, t)
    y.sum().backward()

    assert x.grad is not None
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"no grad for parameter {name}"


@pytest.mark.parametrize("t_factory", [
    lambda B: torch.tensor(0.5),
    lambda B: torch.full((B, 1), 0.5),
])
def test_t_broadcast_shapes(t_factory):
    torch.manual_seed(0)
    layer = SphericalConvModulated(
        2, 3, (6, 6), factorization="dense",
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod("real"),
    )
    B = 2
    x = torch.randn(B, 2, 12, 12)
    t = t_factory(B)
    if t.ndim == 0:
        t = t.expand(B, 1)
    y = layer(x, t)
    assert y.shape == (B, 3, 12, 12)


def test_enabled_flag_false_is_inert():
    """mode_modulation={'enabled': False, ...} behaves like no modulation."""
    layer = SphericalConvModulated(
        2, 3, (6, 6), factorization="dense",
        embed=_default_embed(dim=8),
        mode_modulation={"enabled": False, "type": "real",
                         "hidden_channels": 16, "full_res": False},
    )
    assert layer.modulator is None
    x = torch.randn(2, 2, 12, 12)
    y = layer(x)  # no t needed
    assert y.shape == (2, 3, 12, 12)


def test_missing_t_when_enabled_raises():
    layer = SphericalConvModulated(
        2, 3, (6, 6), factorization="dense",
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod("real"),
    )
    x = torch.randn(2, 2, 12, 12)
    with pytest.raises(ValueError, match="t"):
        layer(x)


def test_modulation_without_embed_raises():
    with pytest.raises(ValueError, match="embed"):
        SphericalConvModulated(
            2, 3, (6, 6), factorization="dense",
            embed=None,
            mode_modulation=_default_mod("real"),
        )


def test_unknown_modulation_type_raises():
    with pytest.raises(ValueError, match="mode_modulation"):
        SphericalConvModulated(
            2, 3, (6, 6), factorization="dense",
            embed=_default_embed(dim=8),
            mode_modulation={"enabled": True, "type": "bogus",
                             "hidden_channels": 16, "full_res": False},
        )


def test_unknown_embed_type_raises():
    with pytest.raises(ValueError, match="type_t"):
        SphericalConvModulated(
            2, 3, (6, 6), factorization="dense",
            embed={"type_t": "bogus", "type_k": "power", "dim": 8,
                   "alpha": -2.0, "r": 10000.0},
            mode_modulation=_default_mod("real"),
        )


def test_share_m_changes_modulator_in_features():
    """share_m flips the modulator MLP's input width."""
    layer_share = SphericalConvModulated(
        2, 3, (6, 6), factorization="dense",
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod("real", share_m=True),
    )
    layer_full = SphericalConvModulated(
        2, 3, (6, 6), factorization="dense",
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod("real", share_m=False),
    )
    # share_m=True: 1 axis (l) + t → 2 * D
    # share_m=False: 2 axes (l, m) + t → 3 * D
    assert layer_share.modulator.in_channels == 8 * 2
    assert layer_full.modulator.in_channels == 8 * 3
