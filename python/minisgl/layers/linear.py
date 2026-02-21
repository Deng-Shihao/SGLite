from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP, _STATE_DICT, _concat_prefix
from .quantization import LinearMethodBase, UnquantizedLinearMethod


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism and quantization support."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self._linear_method = linear_method or UnquantizedLinearMethod()
        self._weights: Dict[str, Any] = self._linear_method.create_weights(
            local_isize, local_osize
        )
        # Store metadata separately (non-tensor values starting with _)
        self._weight_metadata: Dict[str, Any] = {}
        for name in list(self._weights.keys()):
            if name.startswith("_"):
                self._weight_metadata[name] = self._weights.pop(name)

        # Expose weights as attributes for state_dict compatibility
        # _weights.items()
        #    k       v
        # [weight: weight]
        # self.weight = tensor
        for name, tensor in self._weights.items():
            setattr(self, name, tensor)

        self.bias = torch.empty(local_osize) if has_bias else None
        self._weights_processed = False

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """Load state dict and process weights for quantization methods that need it."""
        # First load weights using parent's method
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)

            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        # Sync _weights dict from loaded attributes
        for name in list(self._weights.keys()):
            if hasattr(self, name):
                self._weights[name] = getattr(self, name)

        # Add metadata back to weights for processing
        for name, value in self._weight_metadata.items():
            self._weights[name] = value

        # Process weights if the linear method requires it (e.g., Marlin repacking)
        if not self._weights_processed:
            processed_weights = self._linear_method.process_weights_after_loading(
                self._weights
            )

            # Update weights dict and attributes with processed values
            self._weight_metadata = {}
            for name, value in processed_weights.items():
                if name.startswith("_"):
                    self._weight_metadata[name] = value
                elif isinstance(value, torch.Tensor):
                    self._weights[name] = value
                    setattr(self, name, value)
                else:
                    # Non-tensor, non-metadata values (like workspace)
                    self._weights[name] = value

            self._weights_processed = True

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sync weight dict with potentially reloaded attributes
        for name in list(self._weights.keys()):
            if hasattr(self, name):
                self._weights[name] = getattr(self, name)
        # Add metadata back for apply_weights
        for name, value in self._weight_metadata.items():
            self._weights[name] = value
        return self._linear_method.apply_weights(self._weights, x, self.bias)


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(
            input_size,
            output_size,
            input_size,
            tp_output_size,
            has_bias,
            linear_method,
        )


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        tp_info = get_tp_info()

        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            linear_method,
        )


class LinearOProj(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = divide_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            linear_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sync weight dict with potentially reloaded attributes
        for name in list(self._weights.keys()):
            if hasattr(self, name):
                self._weights[name] = getattr(self, name)
        # Add metadata back for apply_weights
        for name, value in self._weight_metadata.items():
            self._weights[name] = value
        y = self._linear_method.apply_weights(self._weights, x, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            input_size,
            output_size,
            local_input_size,
            local_output_size,
            has_bias,
            linear_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sync weight dict with potentially reloaded attributes
        for name in list(self._weights.keys()):
            if hasattr(self, name):
                self._weights[name] = getattr(self, name)
        # Add metadata back for apply_weights
        for name, value in self._weight_metadata.items():
            self._weights[name] = value
        y = self._linear_method.apply_weights(self._weights, x, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y

