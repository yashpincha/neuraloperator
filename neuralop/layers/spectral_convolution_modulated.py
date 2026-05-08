"""SpectralConv_modulated: SpectralConv extended with optional time- and
mode-conditioned modulation of spectral coefficients.

The layer is identical to :class:`SpectralConv` when both ``embed`` and
``mode_modulation`` are ``None``; in that case ``forward`` accepts an
optional ``t`` that is ignored. When ``mode_modulation`` is provided, a
small MLP whose input is a learned embedding of ``(t, k)`` produces a
per-mode multiplier that is applied to the spectral coefficients before
the convolution contraction.
"""
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .channel_mlp import ChannelMLP
from .spectral_convolution import SpectralConv

Number = Union[int, float]


class SpectralConv_modulated(SpectralConv):
    """Spectral convolution with optional ``(t, k)`` modulation of modes.

    Subclass of :class:`SpectralConv`. Adds two optional dictionary
    arguments that together turn on modulation; with both set to ``None``
    (the default) the layer is functionally identical to its parent and
    ``forward`` accepts an extra ``t`` argument that is ignored.

    See :class:`SpectralConv` for the inherited parameters.

    Parameters
    ----------
    embed : dict or None, optional
        Configuration for the scalar-time and frequency-mode embeddings.
        Required when ``mode_modulation`` is provided. Keys:

        - ``type_t`` : ``'sinusoidal'`` (default) or ``'power'``.
          Embedding method for the scalar time ``t``.
        - ``type_k`` : ``'power'`` (default) or ``'sinusoidal'``.
          Embedding method for the frequency-mode index ``k``.
        - ``dim`` : int, default 32. Embedding dimension ``D``. Must be
          even when either type is ``'sinusoidal'``.
        - ``alpha`` : float, default ``-2.0``. Exponent range for power
          embedding: features are ``t**p`` with
          ``p in linspace(alpha, 0, D)``.
        - ``r`` : float, default ``10000.0``. Base for the sinusoidal
          embedding frequencies ``r ** (-2i/D)``.

    mode_modulation : dict or None, optional
        Configuration for the per-mode modulation MLP. Keys:

        - ``enabled`` : bool, default ``True``. If ``False`` the layer is
          identical to :class:`SpectralConv`.
        - ``type`` : ``'real'``, ``'complex'``, or ``'polar'``.

          - ``'real'`` -- real-valued multiplier ``h(t, k)``.
          - ``'complex'`` -- complex multiplier ``h_r(t, k) + i h_i(t, k)``.
          - ``'polar'`` -- phase-only rotation ``exp(i theta(t, k))``.

        - ``hidden_channels`` : int, default 64. Hidden width of the
          modulation MLP.
        - ``full_res`` : bool, default ``False``. Reserved for future use;
          currently the MLP always runs at the kept-mode resolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=1.0,
        factorization=None,
        implementation="reconstructed",
        enforce_hermitian_symmetry=True,
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        embed: Optional[dict] = None,
        mode_modulation: Optional[dict] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            bias=bias,
            separable=separable,
            resolution_scaling_factor=resolution_scaling_factor,
            fno_block_precision=fno_block_precision,
            rank=rank,
            factorization=factorization,
            implementation=implementation,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs,
            init_std=init_std,
            fft_norm=fft_norm,
            device=device,
        )

        self._build_embedding(embed)
        self._build_modulator(mode_modulation)

    def _build_embedding(self, embed: Optional[dict]) -> None:
        self.embed_config = embed
        self._k_grid_cache = {}
        if embed is None:
            return

        embed_dim = embed.get("dim", 32)
        alpha = embed.get("alpha", -2.0)
        r = embed.get("r", 10000.0)
        type_t = embed.get("type_t", "sinusoidal")
        type_k = embed.get("type_k", "power")

        # Persist the resolved values back into the config dict so callers
        # can inspect what was actually used.
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

        # Input to modulator: D features for t plus n_dims * D for k.
        in_features = self.embed_dim * (self.order + 1)

        if self.modulation_type in ("real", "polar"):
            mod_out_channels = self.in_channels
        elif self.modulation_type == "complex":
            mod_out_channels = self.in_channels * 2
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
        """Embed scalar time ``t`` and broadcast over a spatial shape.

        Parameters
        ----------
        t : torch.Tensor
            Shape ``(B, 1)`` (or any shape that broadcasts to it).
        shape : tuple of int
            Spatial shape to broadcast to.

        Returns
        -------
        torch.Tensor of shape ``(B, D, *shape)``.
        """
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
        """Embed the per-axis frequency-mode index grid for the kept modes.

        Parameters
        ----------
        shape : tuple of int
            Shape of the kept-mode grid; for real FFT the last axis is
            ``S_N // 2 + 1``.
        device : torch.device or None
            Device for the returned tensor.

        Returns
        -------
        torch.Tensor of shape ``(1, n_dims * D, *shape)``.
        """
        embed_type = self.embed_config["type_k"]
        n_dims = len(shape)

        cache_key = (tuple(shape), device)
        if cache_key not in self._k_grid_cache:
            k_ranges = []
            for i, Si in enumerate(shape):
                if i < n_dims - 1:
                    modes_i = Si // 2
                    k_ranges.append(
                        torch.arange(-modes_i, Si - modes_i, device=device)
                    )
                else:
                    k_ranges.append(torch.arange(0, Si, device=device))
            k_grid = torch.stack(
                torch.meshgrid(*k_ranges, indexing="ij"), dim=0
            ).unsqueeze(0)
            self._k_grid_cache[cache_key] = k_grid
        else:
            k_grid = self._k_grid_cache[cache_key]

        if embed_type == "power":
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
            return torch.complex(
                mlp_out[:, : self.in_channels, ...],
                mlp_out[:, self.in_channels :, ...],
            )
        # 'polar'
        theta = self.modulator(combined)
        return torch.exp(1j * theta)

    def clear_cache(self) -> None:
        """Drop any cached frequency grids."""
        self._k_grid_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        output_shape: Optional[Tuple[int]] = None,
    ):
        """Forward pass of the modulated spectral conv.

        When ``self.modulator is None`` this delegates to the parent
        :class:`SpectralConv` and ``t`` is ignored. Otherwise the spectral
        coefficients of ``x`` are multiplied by the ``(t, k)`` modulation
        factor before the contraction.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, in_channels, d_1, ..., d_N)``.
        t : torch.Tensor, optional
            Scalar time of shape ``(B, 1)``. Required when modulation is
            enabled; ignored otherwise.
        output_shape : tuple of int, optional
            Override the spatial output shape of the inverse FFT.
        """
        if self.modulator is None:
            return super().forward(x, output_shape=output_shape)

        if t is None:
            raise ValueError(
                "SpectralConv_modulated has mode_modulation enabled; "
                "`t` must be provided."
            )

        # ----- Begin copy of SpectralConv.forward, with the single
        # modulation insertion marked below. -----
        batchsize, _, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        if self.fno_block_precision == "mixed":
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size],
            device=x.device,
            dtype=out_dtype,
        )

        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(
                fft_size, self.n_modes, self.max_n_modes
            )
        ]
        if self.separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]
        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [
                slice(None, -starts[-1]) if starts[-1] else slice(None)
            ]

        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]

        weight_start_idx = 1 if self.separable else 2
        slices_x = [slice(None), slice(None)]
        for all_modes, kept_modes in zip(
            fft_size, list(weight.shape[weight_start_idx:])
        ):
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        slices_x = tuple(slices_x)

        # ----- Modulation insertion: multiply the kept spectral
        # coefficients by the (t, k) modulation factor. -----
        kept_shape = tuple(weight.shape[weight_start_idx:])
        t_embed = self.embed_t(t, shape=kept_shape)
        k_embed = self.embed_k(shape=kept_shape, device=x.device)
        mod_factor = self._modulation_factor(t_embed, k_embed)
        x_kept = x[slices_x] * mod_factor
        out_fft[slices_x] = self._contract(
            x_kept, weight, separable=self.separable
        )
        # ----- End modulation insertion. -----

        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                round(s * r)
                for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)
            )

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])

        if self.complex_data:
            x = torch.fft.ifftn(
                out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
            )
        else:
            if self.enforce_hermitian_symmetry:
                out_fft = torch.fft.ifftn(
                    out_fft,
                    s=mode_sizes[:-1],
                    dim=fft_dims[:-1],
                    norm=self.fft_norm,
                )
                out_fft[..., 0].imag.zero_()
                if mode_sizes[-1] % 2 == 0:
                    out_fft[..., -1].imag.zero_()
                x = torch.fft.irfft(
                    out_fft,
                    n=mode_sizes[-1],
                    dim=fft_dims[-1],
                    norm=self.fft_norm,
                )
            else:
                x = torch.fft.irfftn(
                    out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
                )

        if self.bias is not None:
            x = x + self.bias

        return x
