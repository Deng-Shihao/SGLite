# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py
# and https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/marlin_utils.py

from typing import Optional

import numpy
import torch

# Marlin kernel constants
GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
USE_FP32_REDUCE_DEFAULT = True


def get_scalar_types():
    """Get ScalarType and scalar_types from sgl_kernel.

    Returns:
        tuple: (ScalarType, scalar_types)
    """
    try:
        from sgl_kernel.scalar_type import ScalarType, scalar_types

        return ScalarType, scalar_types
    except ImportError:

        class MockScalarType:
            pass

        class MockScalarTypes:
            uint4 = "uint4"
            uint8 = "uint8"

            def __getattr__(self, name: str) -> str:
                return f"mock_{name}"

        return MockScalarType, MockScalarTypes()


ScalarType, scalar_types = get_scalar_types()


def get_device_capability() -> tuple:
    """Get the CUDA device capability.

    Returns (0, 0) if CUDA is not available or not yet initialized,
    to avoid initializing CUDA during config loading.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.cuda.get_device_capability()
    return (0, 0)


def check_marlin_supported(
    quant_type: ScalarType,
    group_size: int,
    has_zp: bool = False,
    device_capability: Optional[int] = None,
    check_device: bool = True,
) -> bool:
    """Check if Marlin supports the given quantization configuration.

    Args:
        quant_type: The quantization type (e.g., scalar_types.uint4)
        group_size: The quantization group size (-1 for channelwise)
        has_zp: Whether zero points are used (True for AWQ)
        device_capability: GPU compute capability (e.g., 80 for A100)
        check_device: Whether to check device capability (set False during config loading)

    Returns:
        bool: True if Marlin supports this configuration
    """
    if check_device:
        if device_capability is None:
            major, minor = get_device_capability()
            device_capability = major * 10 + minor

        # Marlin requires SM80+ (Ampere or newer)
        if device_capability < 80:
            return False

    # Check group size
    if group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
        return False

    # AWQ uses uint4 with zero points
    if has_zp:
        supported_types = [scalar_types.uint4]
    else:
        # GPTQ-style without zero points
        supported_types = [scalar_types.uint4b8, scalar_types.uint8b128]

    return quant_type in supported_types


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    """Verify tensor shapes are compatible with Marlin kernel.

    Args:
        output_size_per_partition: Output dimension per partition
        input_size_per_partition: Input dimension per partition
        input_size: Total input size
        group_size: Quantization group size

    Raises:
        ValueError: If shapes are incompatible with Marlin
    """
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"output_size_per_partition={output_size_per_partition} "
            f"not divisible by {GPTQ_MARLIN_MIN_THREAD_N}. "
            "Consider reducing tensor_parallel_size."
        )

    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"input_size_per_partition={input_size_per_partition} "
            f"not divisible by {GPTQ_MARLIN_MIN_THREAD_K}. "
            "Consider reducing tensor_parallel_size."
        )

    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"input_size_per_partition={input_size_per_partition} "
            f"not divisible by group_size={group_size}. "
            "Consider reducing tensor_parallel_size."
        )


def marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 1) -> torch.Tensor:
    """Create workspace tensor for Marlin kernel.

    Args:
        device: The CUDA device
        max_blocks_per_sm: Maximum blocks per SM

    Returns:
        Workspace tensor of shape [sms * max_blocks_per_sm]
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    """Create empty g_idx tensor (required but not used for AWQ).

    Args:
        device: The CUDA device

    Returns:
        Empty tensor of shape [0]
    """
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def get_scale_perms() -> tuple:
    """Get permutation indices for Marlin scale transformation.

    Returns:
        tuple: (scale_perm, scale_perm_single)
    """
    scale_perm: list = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    """Permute scales from AWQ format to Marlin format.

    Args:
        s: Scales tensor of shape [num_groups, size_n]
        size_k: Input dimension
        size_n: Output dimension
        group_size: Quantization group size

    Returns:
        Permuted scales tensor
    """
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def get_pack_factor(num_bits: int) -> int:
    """Get pack factor for given bit width.

    Args:
        num_bits: Number of bits per weight

    Returns:
        Number of weights packed into int32
    """
    assert 32 % num_bits == 0, f"32 must be divisible by num_bits={num_bits}"
    return 32 // num_bits


def pack_cols(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    """Pack unpacked weights into int32.

    Args:
        q_w: Unpacked weights of shape [size_k, size_n]
        num_bits: Number of bits per weight
        size_k: Input dimension
        size_n: Output dimension

    Returns:
        Packed weights of shape [size_k, size_n // pack_factor]
    """
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res.contiguous()


def unpack_cols(
    packed_q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int
) -> torch.Tensor:
    """Unpack int32 packed weights.

    Args:
        packed_q_w: Packed weights of shape [size_k, size_n // pack_factor]
        num_bits: Number of bits per weight
        size_k: Input dimension
        size_n: Output dimension

    Returns:
        Unpacked weights of shape [size_k, size_n]
    """
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed_q_w.shape == (
        size_k,
        size_n // pack_factor,
    ), f"packed_q_w.shape={packed_q_w.shape}, expected ({size_k}, {size_n // pack_factor})"

    orig_device = packed_q_w.device

    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        vals = packed_q_w_cpu & mask
        packed_q_w_cpu >>= num_bits
        q_res[:, i::pack_factor] = vals

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res.contiguous()


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    """Transform zero points to Marlin format.

    Args:
        zp: Zero points tensor (unpacked)
        size_k: Number of groups
        size_n: Output dimension
        num_bits: Number of bits per weight

    Returns:
        Transformed and packed zero points
    """
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    """Transform AWQ zero points to Marlin format.

    AWQ zero-points are quantized and packed on the column dim.
    In addition, the values are permuted based on dequantizer.
    Here we undo both of these, and then apply marlin permutation
    and pack it back.

    Args:
        q_zp_packed: AWQ packed zero points
        size_k: Number of groups
        size_n: Output dimension
        num_bits: Number of bits per weight

    Returns:
        Zero points in Marlin format
    """
    # Unpack from AWQ column-packed format
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo AWQ interleaving (use argsort to get inverse permutation)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    # Apply Marlin permutation and packing
    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits)
    return marlin_zp


def apply_awq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    quant_type: ScalarType,
    output_size_per_partition: int,
    input_size_per_partition: int,
    bias: Optional[torch.Tensor] = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    """Apply AWQ Marlin linear operation.

    Args:
        input: Input tensor of shape [..., input_size]
        weight: Marlin-format quantized weights
        weight_scale: Marlin-format scales
        weight_zp: Marlin-format zero points
        g_idx: Group indices (empty for AWQ)
        g_idx_sort_indices: Sort indices (empty for AWQ)
        workspace: Workspace tensor
        quant_type: Quantization type
        output_size_per_partition: Output dimension
        input_size_per_partition: Input dimension
        bias: Optional bias tensor
        use_fp32_reduce: Whether to use FP32 for reduction

    Returns:
        Output tensor of shape [..., output_size]
    """
    from sgl_kernel import gptq_marlin_gemm

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    output = gptq_marlin_gemm(
        reshaped_x,
        None,  # c_or_none (optional output accumulator)
        weight,
        weight_scale,
        None,  # global_scale_or_none
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        quant_type,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=False,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)
