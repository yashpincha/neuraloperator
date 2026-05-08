"""SphericalConvModulated: SphericalConv extended with optional time- and
spherical-mode-conditioned modulation of harmonic coefficients.

Identical to :class:`SphericalConv` when both ``embed`` and
``mode_modulation`` are ``None``; in that case ``forward`` accepts an
optional ``t`` that is ignored. When ``mode_modulation`` is provided, a
small MLP whose input is a learned embedding of ``(t, l, m)`` produces a
per-mode multiplier that is applied to the harmonic coefficients before
or after the convolution contraction.
"""
from typing import List, Optional, Tuple, Union

import torch

from .channel_mlp import ChannelMLP
from .spherical_convolution import SphericalConv

Number = Union[int, float]


class SphericalConvModulated(SphericalConv):
    """Spherical convolution with optional ``(t, l, m)`` modulation of modes.

    Subclass of :class:`SphericalConv`. Adds two optional dictionary
    arguments that together turn on modulation; with both set to ``None``
    (the default) the layer is functionally identical to its parent and
    ``forward`` accepts an extra ``t`` argument that is ignored.

    See :class:`SphericalConv` for the inherited parameters.

    Parameters
    ----------
    embed : dict or None, optional
        Configuration for the scalar-time and harmonic-mode embeddings.
        Required when ``mode_modulation`` is provided. Keys:

        - ``type_t`` : ``'sinusoidal'`` (default) or ``'power'``.
        - ``type_k`` : ``'power'`` (default) or ``'sinusoidal'``. Applied
          to harmonic indices ``(l, m)``.
        - ``dim`` : int, default 32. Embedding dimension ``D``.
        - ``alpha`` : float, default ``-2.0``. Power-embedding exponent
          range; features are ``k**p`` with ``p in linspace(alpha, 0, D)``.
        - ``r`` : float, default ``10000.0``. Sinusoidal-embedding base.

    mode_modulation : dict or None, optional
        Configuration for the per-mode modulation MLP. Keys:

        - ``enabled`` : bool, default ``True``. If ``False`` the layer is
          identical to :class:`SphericalConv`.
        - ``type`` : ``'real'``, ``'complex'``, or ``'polar'``.
        - ``hidden_channels`` : int, default 64.
        - ``full_res`` : bool, default ``False``. Reserved for future use.
        - ``share_m`` : bool, default ``True``. If ``True`` the modulation
          depends only on degree ``l`` and is shared across orders ``m``,
          matching :class:`SphericalConv`'s weight sharing across ``m``.
          If ``False`` each ``(l, m)`` mode gets its own modulation.
        - ``pre_modulate`` : bool, default ``True``. If ``True`` apply
          modulation before the contraction, otherwise after.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization="cp",
        implementation="reconstructed",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs=dict(),
        init_std="auto",
        sht_norm="ortho",
        sht_grids="equiangular",
        device=None,
        dtype=torch.float32,
        complex_data=False,  # dummy param until we unify dtype interface
        embed: Optional[dict] = None,
        mode_modulation: Optional[dict] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            max_n_modes=max_n_modes,
            bias=bias,
            separable=separable,
            resolution_scaling_factor=resolution_scaling_factor,
            fno_block_precision=fno_block_precision,
            rank=rank,
            factorization=factorization,
            implementation=implementation,
            fixed_rank_modes=fixed_rank_modes,
            joint_factorization=joint_factorization,
            decomposition_kwargs=decomposition_kwargs,
            init_std=init_std,
            sht_norm=sht_norm,
            sht_grids=sht_grids,
            device=device,
            dtype=dtype,
            complex_data=complex_data,
        )

        self._build_embedding(embed)
        self._build_modulator(mode_modulation)

    def _build_embedding(self, embed: Optional[dict]) -> None:
        self.embed_config = embed
        self._k_grid_cache = {}
        if embed is None:
            return

        embed_dim = int(embed.get("dim", 32))
        alpha = embed.get("alpha", -2.0)
        r = embed.get("r", 10000.0)
        type_t = embed.get("type_t", "sinusoidal")
        type_k = embed.get("type_k", "power")

        self.embed_config["dim"] = embed_dim
        self.embed_config["alpha"] = alpha
        self.embed_config["r"] = r
        self.embed_config["type_t"] = type_t
        self.embed_config["type_k"] = type_k

        self.embed_dim = embed_dim

        if type_t == "power":
            self.register_buffer(
                "t_powers", torch.linspace(alpha, 0.0, embed_dim)
            )
        elif type_t == "sinusoidal":
            indices = torch.arange(0, embed_dim // 2, dtype=torch.float32)
            self.register_buffer(
                "t_inv_freqs", r ** (-2.0 * indices / embed_dim)
            )
        else:
            raise ValueError(f"Unknown embed['type_t']: {type_t!r}")

        if type_k == "power":
            self.register_buffer(
                "k_powers", torch.linspace(alpha, 0.0, embed_dim)
            )
        elif type_k == "sinusoidal":
            indices = torch.arange(0, embed_dim // 2, dtype=torch.float32)
            self.register_buffer(
                "k_inv_freqs", r ** (-2.0 * indices / embed_dim)
            )
        else:
            raise ValueError(f"Unknown embed['type_k']: {type_k!r}")

    def _build_modulator(self, mode_modulation: Optional[dict]) -> None:
        self.mode_modulation_config = mode_modulation
        if mode_modulation is None or not mode_modulation.get("enabled", True):
            self.modulator = None
            return

        if self.embed_config is None:
            raise ValueError(
                "mode_modulation is enabled but `embed` is None. "
                "Both must be provided to enable mode modulation."
            )

        self.modulation_type = mode_modulation.get("type")
        self.modulation_hidden_channels = mode_modulation.get(
            "hidden_channels", 64
        )
        self.modulation_full_res = mode_modulation.get("full_res", False)
        self.share_m = bool(mode_modulation.get("share_m", True))
        self.pre_modulate = bool(mode_modulation.get("pre_modulate", True))

        # Input to modulator: D features for t, plus k features for one
        # axis (share_m=True) or both axes (share_m=False).
        n_k_axes = 1 if self.share_m else self.order
        in_features = self.embed_dim * (1 + n_k_axes)

        # The modulation factor multiplies the spectral coefficients: with
        # pre_modulate=True it acts on the input (in_channels); with
        # pre_modulate=False it acts on the contraction output
        # (out_channels). Size the MLP accordingly.
        self.mod_target_channels = (
            self.in_channels if self.pre_modulate else self.out_channels
        )

        if self.modulation_type in ("real", "polar"):
            mod_out_channels = self.mod_target_channels
        elif self.modulation_type == "complex":
            mod_out_channels = self.mod_target_channels * 2
        else:
            raise ValueError(
                f"Unknown mode_modulation['type']: {self.modulation_type!r}"
            )

        self.modulator = ChannelMLP(
            in_channels=in_features,
            out_channels=mod_out_channels,
            hidden_channels=self.modulation_hidden_channels,
            n_dim=self.order,
        )

    def embed_t(self, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Embed scalar time ``t`` and broadcast over a spectral shape."""
        embed_type = self.embed_config["type_t"]
        batch_size = t.shape[0]

        if embed_type == "power":
            t_embed = (t.clamp_min(1) ** self.t_powers.unsqueeze(0)) * (t > 0)
        else:  # 'sinusoidal'
            t_scaled = t * self.t_inv_freqs.unsqueeze(0)
            t_embed = torch.cat([torch.sin(t_scaled), torch.cos(t_scaled)], dim=-1)

        return t_embed.reshape(batch_size, -1, *([1] * len(shape))).expand(
            batch_size, -1, *shape
        )

    def embed_k(
        self, shape: Tuple[int, ...], device=None
    ) -> torch.Tensor:
        """Embed harmonic-mode indices for the kept ``(l, m)`` coefficients.

        Parameters
        ----------
        shape : tuple of int
            Spectral shape ``(L, M)`` of the kept harmonic coefficients.
        device : torch.device or None.

        Returns
        -------
        torch.Tensor of shape ``(1, n_k_axes * D, L, M)`` where
        ``n_k_axes`` is 1 if ``share_m`` else 2.
        """
        embed_type = self.embed_config["type_k"]
        n_dims = len(shape)
        cache_key = (tuple(shape), bool(self.share_m), device)

        if cache_key not in self._k_grid_cache:
            if self.share_m:
                # Single l-axis grid, broadcast across m.
                l_grid = (
                    torch.arange(0, shape[0], device=device)
                    .reshape(1, 1, shape[0], 1)
                    .expand(1, 1, shape[0], shape[1])
                )
                k_grid = l_grid
            else:
                k_ranges = [
                    torch.arange(0, shape[0], device=device),
                    torch.arange(0, shape[1], device=device),
                ]
                k_grid = torch.stack(
                    torch.meshgrid(*k_ranges, indexing="ij"), dim=0
                ).unsqueeze(0)
            self._k_grid_cache[cache_key] = k_grid
        else:
            k_grid = self._k_grid_cache[cache_key]

        if embed_type == "power":
            # Spherical-harmonic indices (l, m) are non-negative; sign is
            # always +1, but kept for code symmetry with the FFT version.
            sign = torch.sign(k_grid)
            k_embed = sign.unsqueeze(2) * (
                k_grid.abs().clamp_min(1.0).unsqueeze(2)
                ** self.k_powers.view(1, 1, -1, *([1] * n_dims))
            )
        else:  # 'sinusoidal'
            k_scaled = k_grid.unsqueeze(2) * self.k_inv_freqs.view(
                1, 1, -1, *([1] * n_dims)
            )
            k_embed = torch.cat([torch.sin(k_scaled), torch.cos(k_scaled)], dim=2)

        return k_embed.reshape(1, -1, *shape)

    def _modulation_factor(
        self, t_feature: torch.Tensor, k_feature: torch.Tensor
    ) -> torch.Tensor:
        batch_size = t_feature.shape[0]
        spatial_shape = t_feature.shape[2:]

        combined = torch.cat(
            [t_feature, k_feature.expand(batch_size, -1, *spatial_shape)],
            dim=1,
        )

        if self.modulation_type == "real":
            return self.modulator(combined)
        if self.modulation_type == "complex":
            mlp_out = self.modulator(combined)
            n = self.mod_target_channels
            return torch.complex(mlp_out[:, :n, ...], mlp_out[:, n:, ...])
        # 'polar'
        theta = self.modulator(combined)
        return torch.exp(1j * theta)

    def clear_cache(self) -> None:
        """Drop any cached harmonic-mode grids."""
        self._k_grid_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        output_shape: Optional[Tuple[int]] = None,
    ):
        """Forward pass of the modulated spherical conv.

        When ``self.modulator is None`` this delegates to the parent
        :class:`SphericalConv` and ``t`` is ignored. Otherwise the
        harmonic coefficients are multiplied by the ``(t, l, m)``
        modulation factor before (or after) the contraction.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_channels, nlat, nlon)``.
        t : torch.Tensor, optional
            Scalar time of shape ``(B, 1)``. Required when modulation is
            enabled; ignored otherwise.
        output_shape : tuple of int, optional
            Target ``(nlat, nlon)`` of the inverse SHT.
        """
        if self.modulator is None:
            return super().forward(x, output_shape=output_shape)

        if t is None:
            raise ValueError(
                "SphericalConvModulated has mode_modulation enabled; "
                "`t` must be provided."
            )

        _, _, height, width = x.shape

        if self.resolution_scaling_factor is not None and output_shape is None:
            scaling_factors = self.resolution_scaling_factor
            height = round(height * scaling_factors[0])
            width = round(width * scaling_factors[1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]

        out_fft = self.sht_handle.sht(
            x,
            s=(self.n_modes[0], self.n_modes[1] // 2),
            norm=self.sht_norm,
            grid=self.sht_grids[0],
        )

        # Recent torch-harmonics applies triangular truncation that can
        # return fewer modes than requested. Clamp to the actual SHT
        # output shape so the modulation factor and weight slice all
        # share the same spectral dimensions before contraction.
        modes_height = min(self.n_modes[0], out_fft.shape[-2])
        modes_width = min(self.n_modes[1] // 2, out_fft.shape[-1])
        out_fft = out_fft[:, :, :modes_height, :modes_width]

        kept_shape = (modes_height, modes_width)
        t_embed = self.embed_t(t, shape=kept_shape)
        k_embed = self.embed_k(shape=kept_shape, device=x.device)
        mod_factor = self._modulation_factor(t_embed, k_embed)

        if self.pre_modulate:
            out_fft = out_fft * mod_factor
        out_fft = self._contract(
            out_fft,
            self.weight[:, :, :modes_height],
            separable=self.separable,
            dhconv=True,
        )
        if not self.pre_modulate:
            out_fft = out_fft * mod_factor

        x = self.sht_handle.isht(
            out_fft,
            s=(height, width),
            norm=self.sht_norm,
            grid=self.sht_grids[1],
        )

        if self.bias is not None:
            x = x + self.bias

        return x
