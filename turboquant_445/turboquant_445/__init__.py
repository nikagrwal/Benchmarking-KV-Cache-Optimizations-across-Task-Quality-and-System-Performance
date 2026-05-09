"""TurboQuant compatibility package for Transformers 4.45.2."""

from turboquant_445.cache import (
    TQLayerFused,
    TurboQuantCache,
    TurboQuantLayerState,
    detect_outlier_channels,
    effective_bit_width,
    get_baseline_kv_memory,
)
from turboquant_445.core import (
    QJL,
    TurboQuantConfig,
    TurboQuantMSE,
    TurboQuantProd,
    compute_inner_product_error,
    compute_memory_bytes,
    compute_mse,
)
from turboquant_445.packing import compression_ratio, pack_indices, packed_size_bytes, unpack_indices

__all__ = [
    "TurboQuantConfig",
    "TurboQuantMSE",
    "TurboQuantProd",
    "QJL",
    "TurboQuantCache",
    "TurboQuantLayerState",
    "TQLayerFused",
    "detect_outlier_channels",
    "effective_bit_width",
    "get_baseline_kv_memory",
    "compute_mse",
    "compute_inner_product_error",
    "compute_memory_bytes",
    "pack_indices",
    "unpack_indices",
    "packed_size_bytes",
    "compression_ratio",
]
