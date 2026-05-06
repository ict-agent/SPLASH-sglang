import torch
import numpy as np
from sgl_kernel import moe_fused_gate
from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

def test_moe_fused_gate():
    # Test parameters
    num_tokens = 4
    num_experts = 160  # 96 experts, multiple of 32
    num_expert_group = 1  # 32 groups
    topk_group = 1  # Select top 8 groups
    topk = 9  # Select top 6 experts
    num_fused_shared_experts = 1

    # Create test data
    torch.manual_seed(42)
    input_tensor = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda')
    bias = torch.randn(num_experts, dtype=torch.float32, device='cuda')

    print(f"Input shape: {input_tensor.shape}")
    print(f"Bias shape: {bias.shape}")
    print(f"Num experts: {num_experts}")
    print(f"Num expert groups: {num_expert_group}")
    print(f"Topk groups: {topk_group}")
    print(f"Topk: {topk}")

    # Test our kernel
    try:
        print("\n=== Testing our moe_fused_gate kernel ===")
        weights_kernel, indices_kernel = moe_fused_gate(
            input_tensor,
            bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            1.0,  # routed_scaling_factor
            False  # apply_routed_scaling_factor_on_output
        )
        print(f"Kernel weights shape: {weights_kernel.shape}")
        print(f"Kernel indices shape: {indices_kernel.shape}")
        print(f"Kernel weights dtype: {weights_kernel.dtype}")
        print(f"Kernel indices dtype: {indices_kernel.dtype}")
        print(f"Kernel weights:\n{weights_kernel}")
        print(f"Kernel indices:\n{indices_kernel}")
        print("Kernel executed successfully!")

    except Exception as e:
        print(f"Kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Python implementation
    try:
        print("\n=== Testing Python implementation ===")
        weights_python, indices_python = biased_grouped_topk_impl(
            input_tensor,
            input_tensor,
            bias,
            topk,
            renormalize=True,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=1.0,
            apply_routed_scaling_factor_on_output=False
        )
        print(f"Python weights shape: {weights_python.shape}")
        print(f"Python indices shape: {indices_python.shape}")
        print(f"Python weights dtype: {weights_python.dtype}")
        print(f"Python indices dtype: {indices_python.dtype}")
        print(f"Python weights:\n{weights_python}")
        print(f"Python indices:\n{indices_python}")
        print("Python implementation executed successfully!")

    except Exception as e:
        print(f"Python implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare results
    print("\n=== Comparing results ===")

    # Check shapes first
    if weights_kernel.shape != weights_python.shape:
        print(f"Weights shape mismatch: kernel {weights_kernel.shape} vs python {weights_python.shape}")
        return False
    if indices_kernel.shape != indices_python.shape:
        print(f"Indices shape mismatch: kernel {indices_kernel.shape} vs python {indices_python.shape}")
        return False

    print(f"Shapes match: {weights_kernel.shape}, {indices_kernel.shape}")

    # Check dtypes
    if weights_kernel.dtype != weights_python.dtype:
        print(f"Weights dtype mismatch: kernel {weights_kernel.dtype} vs python {weights_python.dtype}")
    if indices_kernel.dtype != indices_python.dtype:
        print(f"Indices dtype mismatch: kernel {indices_kernel.dtype} vs python {indices_python.dtype}")

    # Compare values
    weights_diff = torch.abs(weights_kernel - weights_python).max()
    indices_diff = torch.abs(indices_kernel.float() - indices_python.float()).max()

    print(f"Max weights difference: {weights_diff}")
    print(f"Max indices difference: {indices_diff}")

    # Print detailed comparison for first token
    print(f"\nDetailed comparison for first token:")
    print(f"Kernel weights[0]: {weights_kernel[0]}")
    print(f"Python weights[0]: {weights_python[0]}")
    print(f"Kernel indices[0]: {indices_kernel[0]}")
    print(f"Python indices[0]: {indices_python[0]}")

    if weights_diff < 1e-4 and indices_diff < 1e-4:
        print("Results match within tolerance!")
        return True
    else:
        print("Results don't match!")
        return False

if __name__ == "__main__":
    success = test_moe_fused_gate()
    if success:
        print("\nAll tests passed!")
    else:
        print("\nTests failed!")
