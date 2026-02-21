"""Unit tests for AWQ quantization kernel."""
from __future__ import annotations

import torch
from minisgl.layers.quantization.awq_triton import awq_gemm_triton
from minisgl.utils import call_if_main


def pack_int4_to_int32(values: torch.Tensor) -> torch.Tensor:
    """Pack 8 int4 values into a single int32.
    
    AWQ uses a specific ordering for the 8 int4 values within each int32.
    The order is: [0, 4, 1, 5, 2, 6, 3, 7] (interleaved pattern).
    """
    assert values.shape[-1] % 8 == 0
    # Reshape to group 8 values together
    reshaped = values.reshape(*values.shape[:-1], -1, 8)
    
    # AWQ interleave order: positions 0,4 come first, then 1,5, then 2,6, then 3,7
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    reordered = reshaped[..., awq_order]
    
    # Pack 8 int4 values into int32
    packed = torch.zeros(*reordered.shape[:-1], dtype=torch.int32, device=values.device)
    for i in range(8):
        packed |= (reordered[..., i].to(torch.int32) & 0xF) << (i * 4)
    
    return packed


def unpack_int32_to_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 to 8 int4 values (reverse of pack_int4_to_int32)."""
    result = torch.zeros(*packed.shape, 8, dtype=torch.int32, device=packed.device)
    for i in range(8):
        result[..., i] = (packed >> (i * 4)) & 0xF
    
    # Reverse AWQ order
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order = [awq_order.index(i) for i in range(8)]
    result = result[..., reverse_order]
    
    return result.reshape(*packed.shape[:-1], -1)


def quantize_awq_style(weight: torch.Tensor, group_size: int) -> tuple:
    """Quantize a weight matrix in AWQ style.
    
    Args:
        weight: [K, N] float weight matrix (input_features, output_features)
        group_size: Quantization group size
        
    Returns:
        qweight: [K, N/8] packed int32 quantized weights
        scales: [K/group_size, N] float16 scales
        qzeros: [K/group_size, N/8] packed int32 zeros
    """
    K, N = weight.shape
    assert K % group_size == 0
    assert N % 8 == 0
    
    # Reshape to groups
    num_groups = K // group_size
    weight_grouped = weight.reshape(num_groups, group_size, N)
    
    # Find min/max per group
    w_min = weight_grouped.min(dim=1, keepdim=True)[0]
    w_max = weight_grouped.max(dim=1, keepdim=True)[0]
    
    # Compute scales and zeros for 4-bit quantization (0-15 range)
    scales = (w_max - w_min) / 15.0
    scales = scales.squeeze(1).to(torch.float16)  # [num_groups, N]
    
    # Compute zero points
    zeros = -w_min / scales.unsqueeze(1)
    zeros = zeros.round().clamp(0, 15).to(torch.int32).squeeze(1)  # [num_groups, N]
    
    # Quantize weights
    qweight = torch.zeros(K, N, dtype=torch.int32, device=weight.device)
    for g in range(num_groups):
        start = g * group_size
        end = (g + 1) * group_size
        w_group = weight[start:end, :]
        qweight[start:end, :] = (
            (w_group / scales[g:g+1, :].float()) + zeros[g:g+1, :].float()
        ).round().clamp(0, 15).to(torch.int32)
    
    # Pack weights and zeros
    qweight_packed = pack_int4_to_int32(qweight)  # [K, N/8]
    qzeros_packed = pack_int4_to_int32(zeros)  # [num_groups, N/8]
    
    return qweight_packed, scales, qzeros_packed


def dequantize_awq_reference(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int) -> torch.Tensor:
    """Reference dequantization for verification."""
    # Unpack weights and zeros
    weight_int = unpack_int32_to_int4(qweight)  # [K, N]
    zeros_int = unpack_int32_to_int4(qzeros)    # [num_groups, N]
    
    K, N = weight_int.shape
    num_groups = scales.shape[0]
    
    # Dequantize
    weight = torch.zeros(K, N, dtype=torch.float16, device=qweight.device)
    for g in range(num_groups):
        start = g * group_size
        end = (g + 1) * group_size
        weight[start:end, :] = (
            (weight_int[start:end, :].float() - zeros_int[g:g+1, :].float()) 
            * scales[g:g+1, :].float()
        ).to(torch.float16)
    
    return weight


def test_awq_kernel_correctness():
    """Test AWQ kernel correctness against reference implementation."""
    torch.manual_seed(42)
    
    # Test parameters
    M = 32  # batch size
    K = 256  # input features (must be divisible by group_size)
    N = 512  # output features (must be divisible by 8)
    group_size = 128
    
    print(f"Testing AWQ kernel: M={M}, K={K}, N={N}, group_size={group_size}")
    
    # Create random input and weight
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    weight = torch.randn(K, N, dtype=torch.float32, device="cuda") * 0.1
    
    # Quantize weight
    qweight, scales, qzeros = quantize_awq_style(weight, group_size)
    
    # Reference: dequantize and do matmul
    weight_dequant = dequantize_awq_reference(qweight, scales, qzeros, group_size)
    reference_output = x @ weight_dequant
    
    # AWQ kernel output
    pack_factor = 8
    awq_output = awq_gemm_triton(x, qweight, scales, qzeros, pack_factor)
    
    # Compare outputs
    max_diff = (awq_output - reference_output).abs().max().item()
    mean_diff = (awq_output - reference_output).abs().mean().item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    # Check tolerance (AWQ uses approximate computation, so we allow some error)
    tolerance = 0.1  # Allow 0.1 max difference
    if max_diff < tolerance:
        print("✓ AWQ kernel test PASSED")
        return True
    else:
        print("✗ AWQ kernel test FAILED")
        print(f"  Max diff {max_diff} > tolerance {tolerance}")
        return False


def test_awq_linear_method():
    """Test AWQLinearMethod create_weights and apply_weights."""
    from minisgl.layers.quantization.awq import AWQConfig, AWQLinearMethod
    
    print("\nTesting AWQLinearMethod...")
    
    # Create AWQ config and linear method
    config = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
    linear_method = AWQLinearMethod(config)
    
    # Create weights
    input_size = 256
    output_size = 512
    weights = linear_method.create_weights(input_size, output_size, torch.float16)
    
    # Verify weight shapes
    pack_factor = 8
    assert "qweight" in weights
    assert "qzeros" in weights
    assert "scales" in weights
    
    assert weights["qweight"].shape == (input_size, output_size // pack_factor)
    assert weights["scales"].shape == (input_size // config.group_size, output_size)
    assert weights["qzeros"].shape == (input_size // config.group_size, output_size // pack_factor)
    
    print("  Weight shapes correct:")
    print(f"    qweight: {weights['qweight'].shape}")
    print(f"    scales: {weights['scales'].shape}")
    print(f"    qzeros: {weights['qzeros'].shape}")
    
    # Test apply_weights with dummy data
    M = 16
    x = torch.randn(M, input_size, dtype=torch.float16, device="cuda")
    
    # Initialize weights with some values
    weights["qweight"] = torch.randint(0, 2**32-1, weights["qweight"].shape, dtype=torch.int32, device="cuda")
    weights["scales"] = torch.randn_like(weights["scales"]) * 0.1
    weights["qzeros"] = torch.randint(0, 2**32-1, weights["qzeros"].shape, dtype=torch.int32, device="cuda")
    
    # Apply weights
    output = linear_method.apply_weights(weights, x, bias=None)
    
    assert output.shape == (M, output_size)
    print(f"  Output shape correct: {output.shape}")
    print("✓ AWQLinearMethod test PASSED")
    return True


@call_if_main()
def main():
    """Run all AWQ tests."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping AWQ tests")
        return
    
    print("=" * 50)
    print("Running AWQ Quantization Tests")
    print("=" * 50)
    
    all_passed = True
    
    try:
        all_passed &= test_awq_kernel_correctness()
    except Exception as e:
        print(f"✗ AWQ kernel test FAILED with exception: {e}")
        all_passed = False
    
    try:
        all_passed &= test_awq_linear_method()
    except Exception as e:
        print(f"✗ AWQLinearMethod test FAILED with exception: {e}")
        all_passed = False
    
    print()
    print("=" * 50)
    if all_passed:
        print("All AWQ tests PASSED!")
    else:
        print("Some AWQ tests FAILED!")
    print("=" * 50)
