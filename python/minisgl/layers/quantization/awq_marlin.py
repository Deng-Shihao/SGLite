# AWQ Marlin quantization support for mini-sglang
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/awq.py

from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from .base_config import LinearMethodBase, QuantizationConfig, set_weight_attrs
from .marlin_utils import (
    ScalarType,
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    check_marlin_supported,
    get_scalar_types,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
    scalar_types,
    verify_marlin_supports_shape,
)

# Import sgl_kernel functions
try:
    from sgl_kernel import awq_marlin_repack
except ImportError:
    awq_marlin_repack = None


class AWQMarlinConfig(QuantizationConfig):
    """Config class for AWQ Marlin.

    Uses highly optimized Marlin CUDA kernels for AWQ inference.
    Requires SM80+ (Ampere) GPUs.
    Supports 4-bit and 8-bit quantization with group sizes -1, 32, 64, 128.

    Reference: https://arxiv.org/abs/2306.00978
    """

    # Mapping from weight bits to scalar type
    TYPE_MAP = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // weight_bits

        if weight_bits not in self.TYPE_MAP:
            raise ValueError(
                f"AWQ Marlin only supports {list(self.TYPE_MAP.keys())} bits, "
                f"but got {weight_bits} bits."
            )

        self.quant_type = self.TYPE_MAP[weight_bits]

        # Verify Marlin config compatibility (not device capability - checked later)
        if not check_marlin_supported(
            self.quant_type, group_size, zero_point, check_device=False
        ):
            raise ValueError(
                f"AWQ Marlin does not support this configuration: "
                f"bits={weight_bits}, group_size={group_size}, zero_point={zero_point}. "
                f"group_size must be in [-1, 32, 64, 128]."
            )

    def __repr__(self) -> str:
        return (
            f"AWQMarlinConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point})"
        )

    def get_name(self) -> str:
        return "awq_marlin"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_min_capability(self) -> int:
        # Marlin requires Ampere or newer GPUs
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            "quantize_config.json",  # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQMarlinConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    @classmethod
    def is_awq_marlin_compatible(cls, quant_config: Dict[str, Any]) -> bool:
        """Check if an AWQ config can be upgraded to Marlin.

        Note: This only checks config parameters, not GPU capability.
        GPU capability is checked during weight processing when CUDA is initialized.

        Args:
            quant_config: The quantization config dictionary

        Returns:
            bool: True if compatible with AWQ Marlin
        """
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("w_bit") or quant_config.get("bits")
        group_size = quant_config.get("q_group_size") or quant_config.get("group_size")
        zero_point = quant_config.get("zero_point")

        if quant_method != "awq":
            return False
        if num_bits is None or group_size is None or zero_point is None:
            return False
        if num_bits not in cls.TYPE_MAP:
            return False

        # Only check config parameters, not device capability
        # Device capability is checked during weight processing
        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[num_bits],
            group_size=group_size,
            has_zp=zero_point,
            check_device=False,  # Don't check GPU capability during config loading
        )

    def get_linear_method(self) -> "AWQMarlinLinearMethod":
        return AWQMarlinLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AWQMarlinLinearMethod(LinearMethodBase):
    """Linear method for AWQ Marlin.

    Uses Marlin CUDA kernels for faster inference on Ampere+ GPUs.
    Weights are loaded in AWQ format and repacked to Marlin format
    after loading via process_weights_after_loading().

    Args:
        quant_config: The AWQ Marlin quantization config.
    """

    def __init__(self, quant_config: AWQMarlinConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size: int, output_size: int) -> Dict[str, Any]:
        """Create weight tensors for AWQ Marlin linear layer.

        Weights are created in AWQ format for compatibility with checkpoints.
        They will be repacked to Marlin format after loading.

        Args:
            input_size: Input dimension
            output_size: Output dimension

        Returns:
            Dictionary containing weight tensors and metadata
        """
        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = input_size

        # Verify shape compatibility with Marlin
        verify_marlin_supports_shape(
            output_size_per_partition=output_size,
            input_size_per_partition=input_size,
            input_size=input_size,
            group_size=group_size,
        )

        if input_size % group_size != 0:
            raise ValueError(
                f"input_size={input_size} is not divisible by group_size={group_size}. "
                "This can be caused by too large tensor parallel size."
            )
        if output_size % self.quant_config.pack_factor != 0:
            raise ValueError(
                f"output_size={output_size} is not divisible by pack_factor="
                f"{self.quant_config.pack_factor}. "
                "This can be caused by too large tensor parallel size."
            )

        # Create tensors in AWQ format (will be repacked after loading)
        qweight = Parameter(
            torch.empty(
                input_size,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        num_groups = input_size // group_size

        qzeros = Parameter(
            torch.empty(
                num_groups,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        scales = Parameter(
            torch.empty(
                num_groups,
                output_size,
                device="cuda",
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 0,
                "output_dim": 1,
            },
        )

        return {
            "qweight": qweight,
            "qzeros": qzeros,
            "scales": scales,
            # Metadata for Marlin (stored but not tensors)
            "_input_size": input_size,
            "_output_size": output_size,
            "_num_groups": num_groups,
            "_weights_processed": False,
        }

    def process_weights_after_loading(
        self, weights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Repack weights from AWQ format to Marlin format.

        Called after weights are loaded from checkpoint.
        This transforms the weights for use with the Marlin kernel.

        Args:
            weights: Dictionary containing AWQ-format weights

        Returns:
            Dictionary containing Marlin-format weights
        """
        if weights.get("_weights_processed", False):
            return weights

        if awq_marlin_repack is None:
            raise RuntimeError(
                "sgl_kernel.awq_marlin_repack not available. "
                "Ensure sgl_kernel>=0.3.17.post1 is installed."
            )

        device = weights["qweight"].device
        input_size = weights["_input_size"]
        output_size = weights["_output_size"]
        num_groups = weights["_num_groups"]

        # Create workspace tensor for Marlin kernel
        workspace = marlin_make_workspace(device)

        # Repack qweight from AWQ to Marlin format
        marlin_qweight = awq_marlin_repack(
            weights["qweight"],
            size_k=input_size,
            size_n=output_size,
            num_bits=self.quant_config.quant_type.size_bits,
        )

        # Permute scales to Marlin format
        marlin_scales = marlin_permute_scales(
            weights["scales"],
            size_k=input_size,
            size_n=output_size,
            group_size=self.quant_config.group_size,
        )

        # Transform zero points to Marlin format
        marlin_zp = awq_to_marlin_zero_points(
            weights["qzeros"],
            size_k=num_groups,
            size_n=output_size,
            num_bits=self.quant_config.quant_type.size_bits,
        )

        # Create empty g_idx tensors (not used for AWQ but required by kernel)
        g_idx = marlin_make_empty_g_idx(device)
        g_idx_sort_indices = marlin_make_empty_g_idx(device)

        return {
            "qweight": Parameter(marlin_qweight, requires_grad=False),
            "qzeros": Parameter(marlin_zp, requires_grad=False),
            "scales": Parameter(marlin_scales, requires_grad=False),
            "workspace": workspace,
            "g_idx": g_idx,
            "g_idx_sort_indices": g_idx_sort_indices,
            "_input_size": input_size,
            "_output_size": output_size,
            "_weights_processed": True,
        }

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the quantized weights to the input tensor.

        Args:
            weights: Dictionary containing Marlin-format weights
            x: Input tensor
            bias: Optional bias tensor

        Returns:
            Output tensor
        """
        return apply_awq_marlin_linear(
            input=x,
            weight=weights["qweight"],
            weight_scale=weights["scales"],
            weight_zp=weights["qzeros"],
            g_idx=weights["g_idx"],
            g_idx_sort_indices=weights["g_idx_sort_indices"],
            workspace=weights["workspace"],
            quant_type=self.quant_config.quant_type,
            output_size_per_partition=weights["_output_size"],
            input_size_per_partition=weights["_input_size"],
            bias=bias,
        )
