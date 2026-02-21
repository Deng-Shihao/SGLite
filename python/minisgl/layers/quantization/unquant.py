from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .base_config import LinearMethodBase, set_weight_attrs


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization.

    Args:
        separate_bias_add: If true, add bias separately after matrix
                           multiplication.
    """

    def __init__(self, separate_bias_add: bool = False):
        self.separate_bias_add = separate_bias_add

    def create_weights(self, input_size: int, output_size: int) -> Dict[str, torch.Tensor]:
        weight = Parameter(
            torch.empty(
                output_size,
                input_size,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        return {"weight": weight}

    def apply_weights(
        self,
        weights: Dict[str, torch.Tensor],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        weight = weights["weight"]

        if self.separate_bias_add:
            if bias:
                return F.linear(x, weight) + bias
            return F.linear(x, weight)
        return F.linear(x, weight, bias)
