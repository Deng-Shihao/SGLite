from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
) -> None:
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This is useful
    for recording metadata about the weight tensor, such as its input and
    output dimensions, whether it is packed, and other quantization-related
    information.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing attribute {key} on weight tensor."
        setattr(weight, key, value)


class LinearMethodBase(ABC):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(self, input_size: int, output_size: int) -> Dict[str, torch.Tensor]:
        """Create weights for a linear layer."""
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        weights: Dict[str, torch.Tensor],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the weights to the input tensor."""
        raise NotImplementedError

    def process_weights_after_loading(
        self,
        weights: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process weights after loading from checkpoint.

        Override this method to transform weights (e.g., repack for different format).
        Default implementation returns weights unchanged.

        Args:
            weights: Dictionary containing loaded weight tensors

        Returns:
            Dictionary containing processed weight tensors
        """
        return weights


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    @abstractmethod
    def get_name(self) -> str:
        """Name of the quantization method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @abstractmethod
    def get_min_capability(self) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's quantization config.")

    @abstractmethod
    def get_linear_method(self) -> LinearMethodBase:
        """Get the linear method to use for the quantized linear layer."""
        raise NotImplementedError

    @abstractmethod
    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError
