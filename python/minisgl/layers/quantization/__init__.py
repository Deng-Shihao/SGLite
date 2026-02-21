"""Quantization module for mini-sglang.

Currently supports:
- AWQ (Activation-aware Weight Quantization) using Triton kernels
- AWQ Marlin (AWQ using Marlin CUDA kernels for SM80+ GPUs)
"""

from .awq import AWQConfig, AWQLinearMethod
from .awq_marlin import AWQMarlinConfig, AWQMarlinLinearMethod
from .base_config import LinearMethodBase, QuantizationConfig, set_weight_attrs
from .unquant import UnquantizedLinearMethod

# Map of quantization method names to their config classes
QUANTIZATION_METHODS = {
    "awq": AWQConfig,
    "awq_marlin": AWQMarlinConfig,
}


def get_quantization_config(method_name: str) -> type:
    """Get the quantization config class for a given method name."""
    method_name = method_name.lower()
    if method_name not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Unknown quantization method: {method_name}. "
            f"Supported methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    return QUANTIZATION_METHODS[method_name]


__all__ = [
    "AWQConfig",
    "AWQLinearMethod",
    "AWQMarlinConfig",
    "AWQMarlinLinearMethod",
    "LinearMethodBase",
    "QuantizationConfig",
    "UnquantizedLinearMethod",
    "set_weight_attrs",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
