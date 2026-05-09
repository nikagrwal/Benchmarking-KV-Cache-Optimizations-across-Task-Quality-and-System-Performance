"""
KV cache integration for HuggingFace Transformers 4.45.x using TurboQuant.

This version targets the `DynamicCache` API exposed by transformers 4.45.2,
which stores one key/value tensor per layer and expects cache updates to happen
through `DynamicCache.update(..., layer_idx=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from turboquant_445.core import TurboQuantConfig, TurboQuantMSE
from turboquant_445.packing import pack_indices, packed_size_bytes, unpack_indices


@dataclass
class _QuantizedChunk:
    full_shape: torch.Size
    idx_numel: int
    idx: torch.Tensor
    norms: torch.Tensor
    regular_numel: int | None = None
    regular_idx: torch.Tensor | None = None
    regular_norms: torch.Tensor | None = None
    outlier_numel: int | None = None
    outlier_idx: torch.Tensor | None = None
    outlier_norms: torch.Tensor | None = None



def detect_outlier_channels(key_states: torch.Tensor, num_outliers: int) -> torch.Tensor:
    """Select the highest-RMS channels across batch, heads, and sequence."""
    rms_per_channel = key_states.float().pow(2).mean(dim=(0, 1, 2)).sqrt()
    _, top_indices = rms_per_channel.topk(min(num_outliers, rms_per_channel.shape[0]))
    mask = torch.zeros(rms_per_channel.shape[0], dtype=torch.bool, device=key_states.device)
    mask[top_indices] = True
    return mask



def effective_bit_width(head_dim: int, num_outliers: int, base_bits: int, outlier_bits: int) -> float:
    regular = head_dim - num_outliers
    return (regular * base_bits + num_outliers * outlier_bits) / head_dim


class TurboQuantLayerState:
    """Per-layer compressed storage plus dequantized tensors for HF cache access."""

    def __init__(
        self,
        head_dim: int,
        bit_width: int,
        num_outlier_channels: int = 0,
        outlier_bits: int = 0,
        device: torch.device | None = None,
        use_packing: bool = False,
    ):
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.num_outlier_channels = num_outlier_channels
        self.outlier_bits = outlier_bits
        self.use_packing = use_packing

        dev = device or torch.device("cpu")
        self.regular_dim = head_dim - num_outlier_channels if num_outlier_channels > 0 else head_dim
        self.outlier_dim = num_outlier_channels

        self.quantizer = TurboQuantMSE(
            TurboQuantConfig(bit_width=bit_width, head_dim=self.regular_dim, device=dev)
        )

        if num_outlier_channels > 0 and outlier_bits > bit_width:
            self.outlier_quantizer = TurboQuantMSE(
                TurboQuantConfig(
                    bit_width=outlier_bits,
                    head_dim=self.outlier_dim,
                    device=dev,
                    rotation_seed=43,
                )
            )
        else:
            self.outlier_quantizer = None
            self.regular_dim = head_dim

        self.dtype = torch.float32
        self.device = dev
        self.is_initialized = False
        self._channel_mask: torch.Tensor | None = None
        self._key_chunks: list[_QuantizedChunk] = []
        self._value_chunks: list[_QuantizedChunk] = []
        self._full_keys: torch.Tensor | None = None
        self._full_values: torch.Tensor | None = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True
        if self.outlier_quantizer is not None and self._channel_mask is None:
            self._channel_mask = detect_outlier_channels(key_states, self.num_outlier_channels)

    def _quantize_tensor(self, x: torch.Tensor) -> _QuantizedChunk:
        shape = x.shape
        batch, heads, seq, _ = shape

        if self.outlier_quantizer is not None and self._channel_mask is not None:
            regular_mask = ~self._channel_mask
            x_f = x.float()

            regular = x_f[..., regular_mask]
            outlier = x_f[..., self._channel_mask]

            r_flat = regular.reshape(-1, self.regular_dim)
            r_norms = r_flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            r_idx = self.quantizer.quantize(r_flat / r_norms)

            o_flat = outlier.reshape(-1, self.outlier_dim)
            o_norms = o_flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            o_idx = self.outlier_quantizer.quantize(o_flat / o_norms)

            if self.use_packing:
                r_idx = pack_indices(r_idx.flatten(), self.bit_width)
                o_idx = pack_indices(o_idx.flatten(), self.outlier_bits)

            return _QuantizedChunk(
                full_shape=shape,
                idx_numel=batch * heads * seq * self.head_dim,
                idx=torch.empty(0, dtype=torch.uint8, device=x.device),
                norms=torch.empty(0, dtype=torch.float32, device=x.device),
                regular_numel=batch * heads * seq * self.regular_dim,
                regular_idx=r_idx,
                regular_norms=r_norms.squeeze(-1),
                outlier_numel=batch * heads * seq * self.outlier_dim,
                outlier_idx=o_idx,
                outlier_norms=o_norms.squeeze(-1),
            )

        flat = x.float().reshape(-1, self.head_dim)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        idx = self.quantizer.quantize(flat / norms)

        if self.use_packing:
            idx = pack_indices(idx.flatten(), self.bit_width)

        return _QuantizedChunk(
            full_shape=shape,
            idx_numel=flat.shape[0] * self.head_dim,
            idx=idx,
            norms=norms.squeeze(-1),
        )

    def _dequantize_chunk(self, chunk: _QuantizedChunk) -> torch.Tensor:
        if chunk.regular_idx is not None:
            r_idx = chunk.regular_idx
            o_idx = chunk.outlier_idx

            if self.use_packing:
                r_idx = unpack_indices(r_idx, self.bit_width, chunk.regular_numel).reshape(-1, self.regular_dim)
                o_idx = unpack_indices(o_idx, self.outlier_bits, chunk.outlier_numel).reshape(-1, self.outlier_dim)

            r_hat = self.quantizer.dequantize(r_idx) * chunk.regular_norms.unsqueeze(-1)
            o_hat = self.outlier_quantizer.dequantize(o_idx) * chunk.outlier_norms.unsqueeze(-1)

            result = torch.zeros(chunk.full_shape, dtype=torch.float32, device=self.device)
            regular_mask = ~self._channel_mask
            result[..., regular_mask] = r_hat.reshape(
                chunk.full_shape[0], chunk.full_shape[1], chunk.full_shape[2], self.regular_dim
            )
            result[..., self._channel_mask] = o_hat.reshape(
                chunk.full_shape[0], chunk.full_shape[1], chunk.full_shape[2], self.outlier_dim
            )
            return result.to(self.dtype)

        idx = chunk.idx
        if self.use_packing:
            idx = unpack_indices(idx, self.bit_width, chunk.idx_numel).reshape(-1, self.head_dim)
        flat = self.quantizer.dequantize(idx) * chunk.norms.unsqueeze(-1)
        return flat.reshape(chunk.full_shape).to(self.dtype)

    def _dequantize_all(self, chunks: list[_QuantizedChunk]) -> torch.Tensor:
        if not chunks:
            return torch.tensor([], dtype=self.dtype, device=self.device)
        return torch.cat([self._dequantize_chunk(chunk) for chunk in chunks], dim=-2)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._key_chunks.append(self._quantize_tensor(key_states))
        self._value_chunks.append(self._quantize_tensor(value_states))
        self._full_keys = self._dequantize_all(self._key_chunks)
        self._full_values = self._dequantize_all(self._value_chunks)
        return self._full_keys, self._full_values

    def reset_from_dense(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._key_chunks = []
        self._value_chunks = []
        self._full_keys = None
        self._full_values = None
        self.is_initialized = False
        if keys.numel() == 0 or values.numel() == 0:
            return keys, values
        return self.update(keys, values)

    def get_seq_length(self) -> int:
        if self._full_keys is None or self._full_keys.numel() == 0:
            return 0
        return self._full_keys.shape[-2]

    def get_memory_bytes(self) -> int:
        total_idx_bytes = 0
        total_norm_bytes = 0
        for chunk in self._key_chunks + self._value_chunks:
            if chunk.regular_idx is not None:
                if self.use_packing:
                    total_idx_bytes += chunk.regular_idx.numel() + chunk.outlier_idx.numel()
                else:
                    total_idx_bytes += packed_size_bytes(chunk.regular_numel, self.bit_width)
                    total_idx_bytes += packed_size_bytes(chunk.outlier_numel, self.outlier_bits)
                total_norm_bytes += chunk.regular_norms.numel() * 4
                total_norm_bytes += chunk.outlier_norms.numel() * 4
            else:
                if self.use_packing:
                    total_idx_bytes += chunk.idx.numel()
                else:
                    total_idx_bytes += packed_size_bytes(chunk.idx_numel, self.bit_width)
                total_norm_bytes += chunk.norms.numel() * 4
        return total_idx_bytes + total_norm_bytes

    def get_effective_bits_per_value(self) -> float:
        if self.outlier_quantizer is not None:
            return effective_bit_width(
                self.head_dim,
                self.num_outlier_channels,
                self.bit_width,
                self.outlier_bits,
            )
        return float(self.bit_width)


class TurboQuantCache(DynamicCache):
    """Transformers 4.45-compatible cache that compresses KV states with TurboQuant."""

    def __init__(
        self,
        head_dim: int = 64,
        bit_width: int = 3,
        num_layers: int = 24,
        num_outlier_channels: int = 0,
        outlier_bits: int = 0,
        device: torch.device | None = None,
        use_packing: bool = False,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.bit_width = bit_width
        self.num_outlier_channels = num_outlier_channels
        self.outlier_bits = outlier_bits
        self.device_hint = device
        self.use_packing = use_packing
        self.layers: list[TurboQuantLayerState | None] = [
            TurboQuantLayerState(
                head_dim=head_dim,
                bit_width=bit_width,
                num_outlier_channels=num_outlier_channels,
                outlier_bits=outlier_bits,
                device=device,
                use_packing=use_packing,
            )
            for _ in range(num_layers)
        ]

    def _ensure_layer_slot(self, layer_idx: int) -> TurboQuantLayerState:
        while len(self.layers) <= layer_idx:
            self.layers.append(
                TurboQuantLayerState(
                    head_dim=self.head_dim,
                    bit_width=self.bit_width,
                    num_outlier_channels=self.num_outlier_channels,
                    outlier_bits=self.outlier_bits,
                    device=self.device_hint,
                    use_packing=self.use_packing,
                )
            )
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])
        return self.layers[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        layer = self._ensure_layer_slot(layer_idx)
        keys, values = layer.update(key_states, value_states)
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values
        return keys, values

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx >= len(self.layers) or self.layers[layer_idx] is None:
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> None:
        return None

    def get_memory_bytes(self) -> int:
        return sum(layer.get_memory_bytes() for layer in self.layers if layer is not None)

    def get_effective_bits(self) -> float:
        for layer in self.layers:
            if layer is not None:
                return layer.get_effective_bits_per_value()
        return float(self.bit_width)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.layers)):
            if layer_idx >= len(self.key_cache):
                continue
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            if isinstance(key_tensor, list) or len(key_tensor) == 0:
                continue
            device = key_tensor.device
            beam = beam_idx.to(device)
            reordered_keys = key_tensor.index_select(0, beam)
            reordered_values = value_tensor.index_select(0, beam)
            self.key_cache[layer_idx] = reordered_keys
            self.value_cache[layer_idx] = reordered_values
            self.layers[layer_idx].reset_from_dense(reordered_keys, reordered_values)


class TQLayerFused:
    """Compatibility placeholder kept for API parity with the original package."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TQLayerFused has not been ported to transformers 4.45.2 in turboquant_445. "
            "Use TurboQuantCache for standard generation."
        )



def get_baseline_kv_memory(cache: DynamicCache | tuple) -> int:
    """Calculate the memory footprint of either Cache objects or legacy tuples."""
    total = 0

    if isinstance(cache, tuple):
        for layer_cache in cache:
            if len(layer_cache) != 2:
                continue
            key_tensor, value_tensor = layer_cache
            if key_tensor is None or value_tensor is None or key_tensor.numel() == 0:
                continue
            total += key_tensor.numel() * key_tensor.element_size()
            total += value_tensor.numel() * value_tensor.element_size()
        return total

    for key_tensor, value_tensor in zip(cache.key_cache, cache.value_cache):
        if isinstance(key_tensor, list) or len(key_tensor) == 0:
            continue
        total += key_tensor.numel() * key_tensor.element_size()
        total += value_tensor.numel() * value_tensor.element_size()
    return total
