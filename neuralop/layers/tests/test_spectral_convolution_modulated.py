import pytest
import torch

from ..spectral_convolution import SpectralConv
from ..spectral_convolution_modulated import SpectralConv_modulated


def _default_embed(dim=8):
    return {"type_t": "sinusoidal", "type_k": "power", "dim": dim,
            "alpha": -2.0, "r": 10000.0}


def _default_mod(mod_type="real"):
    return {"enabled": True, "type": mod_type, "hidden_channels": 16,
            "full_res": False}


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("complex_data", [False, True])
def test_no_modulation_matches_parent(dim, complex_data):
    """With embed=mode_modulation=None, output equals SpectralConv on shared weights."""
    torch.manual_seed(0)
    n_modes = (6,) * dim
    in_channels, out_channels = 2, 3
    spatial = (10,) * dim
    dtype = torch.cfloat if complex_data else torch.float32

    base = SpectralConv(
        in_channels, out_channels, n_modes,
        complex_data=complex_data,
    )
    mod = SpectralConv_modulated(
        in_channels, out_channels, n_modes,
        complex_data=complex_data,
        embed=None, mode_modulation=None,
    )
    mod.load_state_dict(base.state_dict())

    x = torch.randn(2, in_channels, *spatial, dtype=dtype)
    with torch.no_grad():
        y_base = base(x)
        y_mod_no_t = mod(x)
        y_mod_with_t = mod(x, t=torch.zeros(2, 1))  # t ignored when modulator is None

    assert torch.allclose(y_base, y_mod_no_t, atol=1e-6)
    assert torch.allclose(y_base, y_mod_with_t, atol=1e-6)
    assert mod.modulator is None


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
@pytest.mark.parametrize("type_t,type_k", [
    ("sinusoidal", "power"),
    ("power", "sinusoidal"),
])
def test_modulated_forward_shape(dim, mod_type, type_t, type_k):
    torch.manual_seed(0)
    n_modes = (6,) * dim
    in_channels, out_channels = 2, 3
    spatial = (10,) * dim

    embed = _default_embed(dim=8)
    embed["type_t"] = type_t
    embed["type_k"] = type_k

    layer = SpectralConv_modulated(
        in_channels, out_channels, n_modes,
        embed=embed, mode_modulation=_default_mod(mod_type),
    )

    x = torch.randn(2, in_channels, *spatial)
    t = torch.tensor([[0.5], [1.5]])
    y = layer(x, t)

    assert y.shape == (2, out_channels, *spatial)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("mod_type", ["real", "complex", "polar"])
def test_modulated_backward_grads_all_params(mod_type):
    torch.manual_seed(0)
    layer = SpectralConv_modulated(
        in_channels=2, out_channels=3, n_modes=(6, 6),
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod(mod_type),
    )
    x = torch.randn(2, 2, 10, 10, requires_grad=True)
    t = torch.tensor([[0.5], [1.5]])

    y = layer(x, t)
    y.sum().backward()

    assert x.grad is not None
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"no grad for parameter {name}"


@pytest.mark.parametrize("t_factory", [
    lambda B: 0.5,                                   # python scalar
    lambda B: torch.tensor(0.5),                     # 0-d tensor
    lambda B: torch.full((B, 1), 0.5),               # (B, 1)
])
def test_t_broadcast_shapes(t_factory):
    torch.manual_seed(0)
    layer = SpectralConv_modulated(
        in_channels=2, out_channels=3, n_modes=(6, 6),
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod("real"),
    )
    B = 2
    x = torch.randn(B, 2, 10, 10)
    t = t_factory(B)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=x.dtype)
    if t.ndim == 0:
        t = t.expand(B, 1)
    elif t.ndim == 1:
        t = t.unsqueeze(-1)
    y = layer(x, t)
    assert y.shape == (B, 3, 10, 10)


def test_enabled_flag_false_is_inert():
    """mode_modulation={'enabled': False, ...} behaves like no modulation."""
    torch.manual_seed(0)
    layer = SpectralConv_modulated(
        in_channels=2, out_channels=3, n_modes=(6, 6),
        embed=_default_embed(dim=8),
        mode_modulation={"enabled": False, "type": "real",
                         "hidden_channels": 16, "full_res": False},
    )
    assert layer.modulator is None
    x = torch.randn(2, 2, 10, 10)
    y = layer(x)  # no t needed
    assert y.shape == (2, 3, 10, 10)


def test_missing_t_when_enabled_raises():
    layer = SpectralConv_modulated(
        in_channels=2, out_channels=3, n_modes=(6, 6),
        embed=_default_embed(dim=8),
        mode_modulation=_default_mod("real"),
    )
    x = torch.randn(2, 2, 10, 10)
    with pytest.raises(ValueError, match="t"):
        layer(x)


def test_modulation_without_embed_raises():
    with pytest.raises(ValueError, match="embed"):
        SpectralConv_modulated(
            in_channels=2, out_channels=3, n_modes=(6, 6),
            embed=None,
            mode_modulation=_default_mod("real"),
        )


def test_unknown_modulation_type_raises():
    with pytest.raises(ValueError, match="mode_modulation"):
        SpectralConv_modulated(
            in_channels=2, out_channels=3, n_modes=(6, 6),
            embed=_default_embed(dim=8),
            mode_modulation={"enabled": True, "type": "bogus",
                             "hidden_channels": 16, "full_res": False},
        )


def test_unknown_embed_type_raises():
    with pytest.raises(ValueError, match="type_t"):
        SpectralConv_modulated(
            in_channels=2, out_channels=3, n_modes=(6, 6),
            embed={"type_t": "bogus", "type_k": "power", "dim": 8,
                   "alpha": -2.0, "r": 10000.0},
            mode_modulation=_default_mod("real"),
        )
