"""
SFNO - Spherical Fourier Neural Operator
Replaces the default SpectralConv (a convolution in the frequency domain
over Fourier basis functions) with a SphericalConv (a convolution over the
spherical harmonic basis functions)
"""

from ..layers.spherical_convolution import SphericalConv
from .fno import (
    FNO,
    partialclass,
    _T_EMB_DEFAULT_EMBED,
    _T_EMB_DEFAULT_MODE_MOD,
)

SFNO = partialclass("SFNO", FNO, factorization="dense", conv_module=SphericalConv)
SFNO.__doc__ = SFNO.__doc__.replace("Fourier", "Spherical Fourier", 1)
SFNO.__doc__ = SFNO.__doc__.replace("FNO", "SFNO")
SFNO.__doc__ = SFNO.__doc__.replace("fno", "sfno")
SFNO.__doc__ = SFNO.__doc__.replace(":ref:`sfno_intro`", ":ref:`fno_intro`")


t_emb_SFNO = partialclass(
    "t_emb_SFNO",
    FNO,
    factorization="dense",
    conv_module=SphericalConv,
    embed=_T_EMB_DEFAULT_EMBED.copy(),
    mode_modulation=_T_EMB_DEFAULT_MODE_MOD.copy(),
)
t_emb_SFNO.__doc__ = t_emb_SFNO.__doc__.replace("Fourier", "Spherical Fourier", 1)
t_emb_SFNO.__doc__ = t_emb_SFNO.__doc__.replace("FNO", "SFNO")
t_emb_SFNO.__doc__ = t_emb_SFNO.__doc__.replace("fno", "sfno")
t_emb_SFNO.__doc__ = t_emb_SFNO.__doc__.replace(":ref:`sfno_intro`", ":ref:`fno_intro`")
