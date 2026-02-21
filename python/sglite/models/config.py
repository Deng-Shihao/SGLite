from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import LlamaConfig

from sglite.utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, float] | None


def load_quantization_config(
    model_path: str,
    hf_config: Optional[Any] = None,
    force_awq_triton: bool = False,
) -> Optional[Any]:
    """Load quantization config from HuggingFace config or model directory.

    Priority:
    1. Check hf_config.quantization_config (from config.json)
    2. Fallback to AWQ config files: quant_config.json or quantize_config.json

    For AWQ models, automatically upgrades to AWQ Marlin on SM80+ GPUs
    unless force_awq_triton is True.

    Args:
        model_path: Path to model or HuggingFace model ID
        hf_config: HuggingFace config object (optional)
        force_awq_triton: If True, force using AWQ Triton instead of Marlin

    Returns:
        The appropriate QuantizationConfig instance or None.
    """
    config_dict = None

    # First, try to get quantization config from HuggingFace config
    if hf_config is not None:
        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is not None:
            # quant_cfg might be a dict or an object with to_dict() method
            if hasattr(quant_cfg, "to_dict"):
                config_dict = quant_cfg.to_dict()
            elif isinstance(quant_cfg, dict):
                config_dict = quant_cfg

    # Fallback: Try to load from config files
    if config_dict is None:
        # Resolve HuggingFace model ID to local cache path if needed
        local_path = model_path
        if not os.path.isdir(model_path):
            try:
                from huggingface_hub import snapshot_download

                # This will return the cached path if already downloaded, or download if not
                local_path = snapshot_download(repo_id=model_path, local_files_only=True)
            except Exception:
                # If it fails (not cached or not a valid repo), return None
                return None

        # Try different AWQ config file names
        for config_name in ["quant_config.json", "quantize_config.json"]:
            config_path = os.path.join(local_path, config_name)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                break

    if config_dict is None:
        return None

    # Detect quantization method and create appropriate config
    quant_method = config_dict.get("quant_method", "").lower()
    is_awq = quant_method == "awq" or "w_bit" in config_dict or "bits" in config_dict

    if is_awq:
        return _create_awq_config(config_dict, force_awq_triton)

    return None


def _create_awq_config(
    config_dict: Dict[str, Any],
    force_awq_triton: bool = False,
) -> Any:
    """Create AWQ config, auto-upgrading to Marlin when possible.

    Args:
        config_dict: The quantization config dictionary
        force_awq_triton: If True, force using AWQ Triton

    Returns:
        AWQMarlinConfig if compatible and not forced, else AWQConfig
    """
    from sglite.layers.quantization import AWQConfig
    from sglite.layers.quantization.awq_marlin import AWQMarlinConfig

    # Try to use AWQ Marlin if compatible and not forced to use Triton
    if not force_awq_triton:
        if AWQMarlinConfig.is_awq_marlin_compatible(config_dict):
            try:
                marlin_config = AWQMarlinConfig.from_config(config_dict)
                logger.info(
                    "Using AWQ Marlin kernel (Marlin CUDA) for quantized inference."
                )
                return marlin_config
            except Exception:
                # Fall back to AWQ Triton if Marlin fails
                pass

    # Fall back to AWQ Triton
    awq_config = AWQConfig.from_config(config_dict)
    logger.info("Using AWQ Triton kernel for quantized inference.")
    return awq_config


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    quantization_config: Optional[Any] = None

    @classmethod
    def from_hf(
        cls,
        config: LlamaConfig,
        model_path: Optional[str] = None,
    ) -> ModelConfig:
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        # Try to load quantization config if model_path is provided
        quant_config = None
        if model_path:
            quant_config = load_quantization_config(model_path, hf_config=config)

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
            quantization_config=quant_config,
        )

    @property
    def is_quantized(self) -> bool:
        """Check if the model is quantized."""
        return self.quantization_config is not None

    @property
    def quant_method_name(self) -> Optional[str]:
        """Get the quantization method name if quantized."""
        if self.quantization_config:
            return self.quantization_config.get_name()
        return None
