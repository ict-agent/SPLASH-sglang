#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <stdio.h>
#include <torch/all.h>

#include <cfloat>
#include <type_traits>
template <typename T, int N>
using AlignedArray = cutlass::AlignedArray<T, N>;
using bfloat16_t = cutlass::bfloat16_t;
using float16_t = cutlass::half_t;
using float32_t = float;

// QQ NOTE: to handle the case for at::Half, error: more than one operator ">" matches these operands: built-in operator
// "arithmetic > arithmetic" function "operator>(const __half &, const __half &)"
template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
  if constexpr (std::is_same<T, at::Half>::value) {
    // at::Half (or float16_t in our native case) causes ambiguity, so we cast to float.
    return static_cast<float>(a) > static_cast<float>(b);
  } else {
    // For types like float, at::BFloat16, or cutlass::half_t / cutlass::bfloat16_t, assume operator> works as expected.
    return a > b;
  }
}

template <typename T>
__device__ inline bool cmp_eq(const T& a, const T& b) {
  if constexpr (std::is_same<T, at::Half>::value) {
    return static_cast<float>(a) == static_cast<float>(b);
  } else {
    return a == b;
  }
}

// Fixed constants common to both dynamic and static template versions:
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int MAX_VPT = 32;  // maximum VPT we support, > params.VPT = num_expert / num_expert_group

// Create an alias for Array using AlignedArray
template <typename T, int N>
using Array = AlignedArray<T, N>;
// QQ: NOTE expression must have a constant value, this has to be > params.VPT
template <typename T>
using AccessType = AlignedArray<T, MAX_VPT>;

template <typename T, typename Params>
__device__ void moe_fused_gate_impl(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    Params params) {
  int tidx = threadIdx.x;
  int64_t thread_row =
      blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
  if (thread_row >= num_rows) {
    return;
  }

  // Calculate topk_excluding_share_expert_fusion from topk
  int64_t topk_excluding_share_expert_fusion = topk - num_fused_shared_experts;

  // Cast pointers to type T:
  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  // Create local arrays for the row chunk and bias chunk and then reinterpret the address of row_chunk as a pointer to
  // AccessType.
  T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> row_chunk;
  AccessType<T> const* vec_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(thread_read_ptr);

  T* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> bias_chunk;
  AccessType<T> const* vec_bias_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(bias_thread_read_ptr);

// QQ NOTE: doing the follow will be slower than loop assign and more importantly
// have misaligned address issue when params.VPT < 8 and mismatch with MAX_VPT
// AccessType<T>* row_chunk_vec_ptr = reinterpret_cast<AccessType<T>*>(&row_chunk);
// row_chunk_vec_ptr[0] = vec_thread_read_ptr[0];
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = vec_thread_read_ptr[0][ii];
    bias_chunk[ii] = vec_bias_thread_read_ptr[0][ii];
  }

  __syncthreads();

////////////////////// Sigmoid //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = static_cast<T>(1.0f / (1.0f + expf(-float(row_chunk[ii]))));
  }
  __syncthreads();

////////////////////// Add Bias //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
  }

////////////////////// Exclude Groups //////////////////////
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group;
       ++k_idx) {  // QQ NOTE Here params.THREADS_PER_ROW = num_expert_group
    int expert = first_elt_read_by_thread;
    // local argmax
    T max_val = static_cast<T>(-FLT_MAX);
    T max_val_second = static_cast<T>(-FLT_MAX);
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      T val = bias_chunk[ii];

      if (cmp_gt(val, max_val)) {
        max_val_second = max_val;
        max_val = val;
      } else if (cmp_gt(val, max_val_second)) {
        max_val_second = val;
      }
    }

    // QQ NOTE: currently fixed to pick top2 sigmoid weight value in each expert group and sum them as the group weight
    // to select expert groups
    T max_sum = max_val + max_val_second;

// argmin reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max_sum =
          static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_sum), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);

      // higher indices win
      if (cmp_gt(max_sum, other_max_sum) || (cmp_eq(other_max_sum, max_sum) && other_expert > expert)) {
        max_sum = other_max_sum;
        expert = other_expert;
      }
    }

    // clear the max value in the thread
    if (k_idx < params.THREADS_PER_ROW - topk_group) {
      int const thread_to_clear_in_group = expert / params.VPT;

      if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
        for (int ii = 0; ii < params.VPT; ++ii) {
          bias_chunk[ii] = static_cast<T>(FLT_MAX);
        }
      }
    }
  }

  __syncthreads();

  ////////////////////// Topk //////////////////////
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
    // local argmax
    T max_val = bias_chunk[0];
    int expert = first_elt_read_by_thread;

    if (!cmp_eq(max_val, static_cast<T>(FLT_MAX))) {
#pragma unroll
      for (int ii = 1; ii < params.VPT; ++ii) {
        T val = bias_chunk[ii];
        if (cmp_gt(val, max_val)) {
          max_val = val;
          expert = first_elt_read_by_thread + ii;
        }
      }
    } else {
      max_val = static_cast<T>(-FLT_MAX);
    }

    // argmax reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max =
          static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_val), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);

      // lower indices to win
      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    int thread_to_clear_in_group = expert / params.VPT;
    int64_t idx = topk * thread_row + k_idx;

    if (thread_group_idx == thread_to_clear_in_group) {
      int expert_to_clear_in_thread = expert % params.VPT;

      // clear the max value in the thread
      bias_chunk[expert_to_clear_in_thread] = static_cast<T>(-FLT_MAX);

      // store output
      output_ptr[idx] = static_cast<float>(row_chunk[expert_to_clear_in_thread]);
      indices_ptr[idx] = static_cast<int32_t>(expert);
    }

    // accumulate sum for all elements
    if (thread_group_idx == 0) {
      output_sum += output_ptr[idx];
    }

    __syncthreads();
  }

  if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
    int64_t last_idx = topk * thread_row + topk_excluding_share_expert_fusion;
    int64_t expert_offset = 0;
    indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);

    // Set the weight to the sum of all weights divided by routed_scaling_factor
    output_ptr[last_idx] = output_sum / routed_scaling_factor;

    if (num_fused_shared_experts > 1) {
      for (int i = 1; i < num_fused_shared_experts; ++i) {
        ++last_idx;
        ++expert_offset;
        indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);
        // Set the weight to the sum of all weights divided by routed_scaling_factor
        output_ptr[last_idx] = output_sum / routed_scaling_factor;
      }
    }
  }
  __syncthreads();

  ////////////////////// Rescale Output //////////////////////
  if (thread_group_idx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t const idx = topk * thread_row + ii;
      output_ptr[idx] = output_ptr[idx] / output_sum;
      if (apply_routed_scaling_factor_on_output) {
        output_ptr[idx] *= routed_scaling_factor;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Templated Kernel Version (using compile-time constants)
//------------------------------------------------------------------------------
template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
  static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
  static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

template <
    typename T,
    int VPT,
    int NUM_EXPERTS,
    int THREADS_PER_ROW,
    int ROWS_PER_WARP,
    int ROWS_PER_CTA,
    int WARPS_PER_CTA>
__global__ void moe_fused_gate_kernel(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> params;
  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      params);
}

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_MOE_GATE_CONFIG(T, EXPERTS, EXPERT_GROUP)                                                 \
  do {                                                                                                   \
    constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                      \
    /* If EXPERT_GROUP > WARP_SIZE, fall back to 1 row per warp */                                       \
    constexpr int ROWS_PER_WARP = ((EXPERT_GROUP) <= WARP_SIZE) ? (WARP_SIZE / (EXPERT_GROUP)) : 1;      \
    constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                          \
    moe_fused_gate_kernel<T, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> \
        <<<num_blocks, block_dim, 0, stream>>>(                                                          \
            input.data_ptr(),                                                                            \
            bias.data_ptr(),                                                                             \
            output.data_ptr<float>(),                                                                    \
            indices.data_ptr<int32_t>(),                                                                 \
            num_rows,                                                                                    \
            topk_group,                                                                                  \
            topk,                                                                                        \
            num_fused_shared_experts,                                                                    \
            routed_scaling_factor,                                                                       \
            apply_routed_scaling_factor_on_output);                                                      \
    dispatched = true;                                                                                   \
  } while (0)

//------------------------------------------------------------------------------
// Dynamic Kernel Version (parameters computed at runtime)
//------------------------------------------------------------------------------
struct KernelParamsDynamic {
  int VPT;
  int NUM_EXPERTS;
  int THREADS_PER_ROW;
  int ROWS_PER_WARP;
  int ROWS_PER_CTA;
  int WARPS_PER_CTA;
};

template <typename T>
__global__ void moe_fused_gate_kernel_dynamic(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  KernelParamsDynamic params;
  params.NUM_EXPERTS = num_experts;             // e.g, for deepseek v3, this is 256
  params.VPT = num_experts / num_expert_group;  // e.g., for deepseek v3, this is 256 / 8 = 32
  params.THREADS_PER_ROW = num_expert_group;    // fixed as num_expert_group, e.g., for deepseek v3, this is 8
  params.WARPS_PER_CTA = WARPS_PER_CTA;         // fixed as 6
  params.ROWS_PER_WARP = std::max<int64_t>(1, WARP_SIZE / num_expert_group);  // WARP_SIZE is fixed as 32
  params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;

  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      params);
}

// Function declarations
template <bool IS_POW2>
inline void moe_fused_gate_vectorized_launcher(
    float const* input,
    float const* bias,
    bool const* finished,
    float* output,
    int32_t* indices,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// Host Launcher Function
//------------------------------------------------------------------------------
std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  TORCH_CHECK(input.dtype() == bias.dtype(), "input and bias should have the same dtype");

  int64_t num_rows = input.size(0);
  int32_t num_experts = input.size(1);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  // Compute grid dimensions based on runtime value for num_expert_group.
  int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / num_expert_group);
  int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
  int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  // GLM NOTE: num_expert_group is 1
  // Prefer generalized vectorized kernel for float32 if conditions fit.
  // This path supports VPT beyond MAX_VPT.
  if (input.scalar_type() == at::kFloat && num_expert_group == 1) {
    if ((num_experts & (num_experts - 1)) == 0) {
      moe_fused_gate_vectorized_launcher<true>(
          input.data_ptr<float>(),
          bias.defined() ? bias.data_ptr<float>() : nullptr,
          /*finished*/ nullptr,
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk,
          num_fused_shared_experts,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output,
          stream);
      return {output, indices};
    }
    if ((num_experts % WARP_SIZE) == 0) {
      // Non-pow2 but multiple of 32: use vectorized m32 kernel
      moe_fused_gate_vectorized_launcher<false>(
          input.data_ptr<float>(),
          bias.defined() ? bias.data_ptr<float>() : nullptr,
          /*finished*/ nullptr,
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk,
          num_fused_shared_experts,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output,
          stream);
      return {output, indices};
    }
  }

  // Check 1: Ensure that num_experts is a power of 2.
  TORCH_CHECK((num_experts & (num_experts - 1)) == 0, "num_experts must be a power of 2, but got ", num_experts);

  // Check 2: Ensure that num_experts is divisible by num_expert_group. (this also means num_expert_group is power of 2)
  TORCH_CHECK(
      num_experts % num_expert_group == 0,
      "num_experts must be divisible by num_expert_group, but got ",
      num_experts,
      " / ",
      num_expert_group);

  int computed_vpt = num_experts / num_expert_group;
  // Check 3: Ensure that num_experts/num_expert_group does not exceed MAX_VPT=32. Maximum VPT indicate max value per
  // threads we can process.
  TORCH_CHECK(
      computed_vpt <= MAX_VPT,
      "Per group experts: num_experts / num_expert_group = (",
      computed_vpt,
      ") exceeds the maximum supported (",
      MAX_VPT,
      ")");

  // Dispatch to templated kernel for known compile-time configurations.
  // We currently only support for:
  //   Case 1: 256 experts, with 8 or 16 groups.
  //   Case 2: 128 experts, with 4 or 8 groups.
  //   Case 3: other cases, require 8 <= num_experts / num_expert_group <= 32
  bool dispatched = false;
  switch (num_experts) {
    case 256:
      if (num_expert_group == 8)
        // This is deepseek v3 case. Here VPT = 256/8 = 32, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 8);
        } else if (num_expert_group == 16)
          // Here VPT = 256/16 = 16, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
          if (input.scalar_type() == at::kBFloat16) {
            LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 16);
          } else if (input.scalar_type() == at::kHalf) {
            LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 16);
          } else if (input.scalar_type() == at::kFloat) {
            LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 16);
          }
      break;
    case 128:
      if (num_expert_group == 4)
        // VPT = 128/4 = 32, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 4);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 4);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 4);
        } else if (num_expert_group == 8)
          // VPT = 128/8 = 16, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
          if (input.scalar_type() == at::kBFloat16) {
            LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 8);
          } else if (input.scalar_type() == at::kHalf) {
            LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 8);
          } else if (input.scalar_type() == at::kFloat) {
            LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 8);
          }
      break;
    default:
      break;
  }
  if (!dispatched) {
    // Fallback to the dynamic kernel if none of the supported combinations match.
    // currently only support num_experts / num_expert_group <= 32 for dynamic kernels
    if (input.scalar_type() == at::kBFloat16) {
      moe_fused_gate_kernel_dynamic<bfloat16_t><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk,
          num_fused_shared_experts,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output);
    } else if (input.scalar_type() == at::kHalf) {
      moe_fused_gate_kernel_dynamic<float16_t><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk,
          num_fused_shared_experts,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output);
    } else if (input.scalar_type() == at::kFloat) {
      moe_fused_gate_kernel_dynamic<float32_t><<<num_blocks, block_dim, 0, stream>>>(
          input.data_ptr(),
          bias.data_ptr(),
          output.data_ptr<float>(),
          indices.data_ptr<int32_t>(),
          num_rows,
          num_experts,
          num_expert_group,
          topk_group,
          topk,
          num_fused_shared_experts,
          routed_scaling_factor,
          apply_routed_scaling_factor_on_output);
    } else {
      TORCH_CHECK(false, "Unsupported data type for moe_fused_gate");
    }
  }
  return {output, indices};
}

// Generalized vectorized kernel for power-of-two num_experts and arbitrary expert groups
template <int VPT, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void moe_fused_gate_vectorized_pow2(
    float const* input,
    float const* bias,
    bool const* finished,
    float* output,
    int64_t const num_rows,
    int32_t* indices,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");
  constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
  static_assert(VPT % ELTS_PER_LDG == 0, "VPT must be multiple of ELTS_PER_LDG");

  // For group=1: each warp (32 threads) processes one row
  int const ELTS_PER_ROW = static_cast<int>(num_experts);
  int const THREADS_PER_ROW = ELTS_PER_ROW / VPT;         // Number of threads needed per row
  int const ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;  // Rows per warp
  int const ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Basic runtime guards
  if ((ELTS_PER_ROW & (ELTS_PER_ROW - 1)) != 0) return;  // power-of-two experts
  if (ELTS_PER_ROW % VPT != 0) return;                   // VPT must divide num_experts
  if (WARP_SIZE % THREADS_PER_ROW != 0) return;          // THREADS_PER_ROW must divide WARP_SIZE

  // Thread mapping: similar to fuse.cu
  int64_t const cta_base_row = blockIdx.x * ROWS_PER_CTA;
  int64_t const warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  int const thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  int64_t const thread_row = warp_base_row + thread_row_in_warp;
  if (thread_row >= num_rows) return;
  if (finished && finished[thread_row]) return;

  // Each thread processes VPT elements with THREADS_PER_ROW spacing
  int const thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  int const first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;

  using AccessType = cutlass::AlignedArray<float, ELTS_PER_LDG>;
  cutlass::Array<float, VPT> row_chunk;
  cutlass::Array<float, VPT> bias_chunk;
  cutlass::Array<float, VPT> sigmoid_chunk;
  AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
  AccessType* bias_chunk_vec_ptr = reinterpret_cast<AccessType*>(&bias_chunk);
  AccessType* sigmoid_chunk_vec_ptr = reinterpret_cast<AccessType*>(&sigmoid_chunk);

  // Memory loading: similar to fuse.cu with THREADS_PER_ROW spacing
  float const* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  float const* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  AccessType const* vec_thread_read_ptr = reinterpret_cast<AccessType const*>(thread_read_ptr);

#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  // Load bias with same pattern
  if (bias != nullptr) {
    AccessType const* vec_bias_read_ptr = reinterpret_cast<AccessType const*>(bias + first_elt_read_by_thread);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      bias_chunk_vec_ptr[ii] = vec_bias_read_ptr[ii * THREADS_PER_ROW];
    }
  } else {
    // Initialize bias to zero when bias is nullptr
    AccessType zero{};
#pragma unroll
    for (int jj = 0; jj < ELTS_PER_LDG; ++jj)
      zero[jj] = 0.f;
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
      bias_chunk_vec_ptr[ii] = zero;
    }
  }

#pragma unroll
  for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
    AccessType row_v = row_chunk_vec_ptr[ldg];
    AccessType bias_v = bias_chunk_vec_ptr[ldg];
    AccessType sig_v;
#pragma unroll
    for (int jj = 0; jj < ELTS_PER_LDG; ++jj) {
      float s = 1.f / (1.f + __expf(-row_v[jj]));
      sig_v[jj] = s;
      row_v[jj] = s + bias_v[jj];
      if (isnan(row_v[jj])) {
        row_v[jj] = 1.0f;
      }
    }
    sigmoid_chunk_vec_ptr[ldg] = sig_v;
    row_chunk_vec_ptr[ldg] = row_v;
  }

  // TopK selection: similar to fuse.cu
  float output_sum = 0.f;
  int start_col = first_elt_read_by_thread;
  int const COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  for (int k_idx = 0; k_idx < topk - num_fused_shared_experts; ++k_idx) {
    float max_val_for_choice = -FLT_MAX;
    float max_val_for_output = 0.f;
    int expert = -1;

    // Find max within this thread's VPT elements using correct indexing
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        int const current_idx_in_chunk = ldg * ELTS_PER_LDG + ii;
        float val_for_choice = row_chunk[current_idx_in_chunk];
        if (val_for_choice > max_val_for_choice) {
          max_val_for_choice = val_for_choice;
          max_val_for_output = sigmoid_chunk[current_idx_in_chunk];
          expert = col + ii;
        }
      }
    }

    // Warp reduction to find global max: similar to fuse.cu
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max_for_choice = __shfl_xor_sync(0xFFFFFFFF, max_val_for_choice, mask, THREADS_PER_ROW);
      float other_max_for_output = __shfl_xor_sync(0xFFFFFFFF, max_val_for_output, mask, THREADS_PER_ROW);
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);
      if (other_max_for_choice > max_val_for_choice ||
          (other_max_for_choice == max_val_for_choice && other_expert < expert)) {
        max_val_for_choice = other_max_for_choice;
        max_val_for_output = other_max_for_output;
        expert = other_expert;
      }
    }

    // Store result (only thread 0 in thread group)
    if (thread_group_idx == 0) {
      int64_t const idx = topk * thread_row + k_idx;
      output[idx] = max_val_for_output;
      indices[idx] = expert;
      output_sum += max_val_for_output;
    }

    // Clear selected expert for next iteration: similar to fuse.cu
    if (k_idx + 1 < topk - num_fused_shared_experts) {
      int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      int const thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

      if (thread_group_idx == thread_to_clear_in_group) {
        int const offset_for_expert = expert % ELTS_PER_LDG;
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -FLT_MAX;
      }
    }
  }

  // Shared expert initialization and final processing
  if (thread_group_idx == 0) {
    // Initialize shared experts with 1/routed_scaling_factor
    if (num_fused_shared_experts > 0) {
      for (int i = 0; i < num_fused_shared_experts; ++i) {
        int64_t const idx = topk * thread_row + (topk - num_fused_shared_experts) + i;
        output[idx] = 1.0f / routed_scaling_factor;
        indices[idx] = static_cast<int32_t>(ELTS_PER_ROW + i);
      }
    }

    // Renormalization and routed_scaling_factor application
    if (output_sum > 0.f) {
      float norm_factor = 1.0f / output_sum;

      // Normalize top-k experts
      for (int k_idx = 0; k_idx < topk - num_fused_shared_experts; ++k_idx) {
        int64_t const idx = topk * thread_row + k_idx;
        output[idx] *= norm_factor;
      }

      // Apply routed_scaling_factor to all experts if needed
      if (apply_routed_scaling_factor_on_output) {
        // Apply to top-k experts
        for (int k_idx = 0; k_idx < topk - num_fused_shared_experts; ++k_idx) {
          int64_t const idx = topk * thread_row + k_idx;
          output[idx] *= routed_scaling_factor;
        }

        // Apply to shared experts (converts 1/routed_scaling_factor to 1.0)
        for (int i = 0; i < num_fused_shared_experts; ++i) {
          int64_t const idx = topk * thread_row + (topk - num_fused_shared_experts) + i;
          output[idx] *= routed_scaling_factor;
        }
      }
    } else {
      // Handle zero sum case
      for (int k_idx = 0; k_idx < topk - num_fused_shared_experts; ++k_idx) {
        int64_t const idx = topk * thread_row + k_idx;
        output[idx] = 0.f;
      }
    }
  }
}

// Generalized vectorized kernel for non-power-of-two num_experts that are multiples of 32
template <int VPT, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void moe_fused_gate_vectorized_m32(
    float const* input,
    float const* bias,
    bool const* finished,
    float* output,
    int64_t const num_rows,
    int32_t* indices,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");
  constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  static_assert(VPT % ELTS_PER_LDG == 0, "VPT must be multiple of ELTS_PER_LDG");

  // Simplified approach: each thread handles ELTS_PER_THREAD elements, 32 threads do butterfly reduction
  constexpr int ELTS_PER_THREAD = VPT / WARP_SIZE;  // e.g., 160/32 = 5
  static_assert(ELTS_PER_THREAD > 0, "ELTS_PER_THREAD must be positive");
  static_assert(VPT % WARP_SIZE == 0, "VPT must be a multiple of WARP_SIZE for this kernel");

  int64_t const cta_base_row = blockIdx.x * WARPS_PER_CTA;
  int64_t const thread_row = cta_base_row + threadIdx.y;
  int const lane_id = threadIdx.x;

  if (thread_row >= num_rows) {
    return;
  }
  if (finished && finished[thread_row]) {
    return;
  }
  float const* thread_row_ptr = input + thread_row * num_experts;
  float const* bias_ptr = bias;

  int base_offset = lane_id * ELTS_PER_THREAD;
  if (base_offset >= num_experts) return;
  float row_chunk[ELTS_PER_THREAD];
  float bias_chunk[ELTS_PER_THREAD];
  float sigmoid_chunk[ELTS_PER_THREAD];
  float choice_chunk[ELTS_PER_THREAD];

  for (int i = 0; i < ELTS_PER_THREAD; i++) {
    int expert_idx = base_offset + i;
    if (expert_idx < num_experts) {
      row_chunk[i] = thread_row_ptr[expert_idx];
      if (isnan(row_chunk[i])) {
        row_chunk[i] = 1.0f;
      }
      bias_chunk[i] = bias_ptr ? bias_ptr[expert_idx] : 0.0f;
      sigmoid_chunk[i] = 1.f / (1.f + __expf(-row_chunk[i]));
      choice_chunk[i] = sigmoid_chunk[i] + bias_chunk[i];  // top-k选择用sigmoid+bias
    }
  }

  float renorm_value = 0.0f;

  // Value-Index pair for topk operations
  struct ValueIndex {
    float value;
    int index;
  };

  // Top-k selection loop
  for (int k_idx = 0; k_idx < topk - num_fused_shared_experts; ++k_idx) {
    // 1. Find local best among ELTS_PER_THREAD elements this thread handles
    ValueIndex thread_local_best;
    thread_local_best.value = -FLT_MAX;
    thread_local_best.index = -1;

    for (int i = 0; i < ELTS_PER_THREAD; i++) {
      int expert_idx = base_offset + i;
      if (expert_idx < num_experts && choice_chunk[i] > thread_local_best.value) {
        thread_local_best.value = choice_chunk[i];
        thread_local_best.index = expert_idx;
      }
    }

    // 2. Warp-level reduction to find global best
    ValueIndex global_best = thread_local_best;
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
      ValueIndex other;
      other.value = __shfl_xor_sync(0xFFFFFFFF, global_best.value, mask);
      other.index = __shfl_xor_sync(0xFFFFFFFF, global_best.index, mask);
      if (other.value > global_best.value || (other.value == global_best.value && other.index < global_best.index)) {
        global_best = other;
      }
    }

    // 3. Broadcast winner's sigmoid value
    int expert = global_best.index;
    float winner_output_val = 0.f;
    if (expert != -1) {
      int winner_thread = expert / ELTS_PER_THREAD;
      int winner_offset = expert % ELTS_PER_THREAD;
      if (winner_thread == lane_id && winner_offset < ELTS_PER_THREAD) {
        winner_output_val = sigmoid_chunk[winner_offset];
      }
      winner_output_val = __shfl_sync(0xFFFFFFFF, winner_output_val, winner_thread);
    }

    // 4. Write results
    if (lane_id == 0) {
      int64_t const idx = topk * thread_row + k_idx;
      output[idx] = winner_output_val;
      indices[idx] = expert;
      renorm_value += winner_output_val;
    }

    // 5. Clear the winner for next iteration
    if (expert != -1) {
      int winner_thread = expert / ELTS_PER_THREAD;
      int winner_offset = expert % ELTS_PER_THREAD;
      if (winner_thread == lane_id && winner_offset < ELTS_PER_THREAD) {
        choice_chunk[winner_offset] = -FLT_MAX;
      }
    }
  }

  // Handle fused shared experts BEFORE normalization
  // Only the first thread in each warp handles shared experts to avoid race conditions
  if (num_fused_shared_experts > 0 && lane_id == 0) {
    for (int i = 0; i < num_fused_shared_experts; i++) {
      int64_t const idx = topk * thread_row + (topk - num_fused_shared_experts) + i;
      // Set shared expert weight to 1/routed_scaling_factor initially
      output[idx] = 1.0f / routed_scaling_factor;
      indices[idx] = num_experts + i;  // This is correct for fused shared experts
    }
  }

  // Renormalization logic - normalize only the top-k selected experts (NOT shared experts)
  if (lane_id == 0) {
    if (renorm_value > 0.f) {
      // Normalize only the top-k selected experts to sum to 1.0
      float norm_factor = 1.0f / renorm_value;

      // Normalize the top-k selected experts
      for (int k_idx = 0; k_idx < topk - num_fused_shared_experts; k_idx++) {
        int64_t const idx = topk * thread_row + k_idx;
        output[idx] *= norm_factor;
        if (apply_routed_scaling_factor_on_output) {
          output[idx] *= routed_scaling_factor;
        }
      }
    }
  }

  // Apply routed_scaling_factor to shared experts based on apply_routed_scaling_factor_on_output
  if (num_fused_shared_experts > 0 && lane_id == 0) {
    for (int i = 0; i < num_fused_shared_experts; i++) {
      int64_t const idx = topk * thread_row + (topk - num_fused_shared_experts) + i;
      if (apply_routed_scaling_factor_on_output) {
        // If applying on output, multiply shared expert weight by routed_scaling_factor
        // This converts 1/routed_scaling_factor to 1.0
        output[idx] *= routed_scaling_factor;
      }
    }
  }
}

template <bool IS_POW2>
inline void moe_fused_gate_vectorized_launcher(
    float const* input,
    float const* bias,
    bool const* finished,
    float* output,
    int32_t* indices,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    cudaStream_t stream) {
  constexpr int BYTES_PER_LDG = 16;
  constexpr int WARPS_PER_TB = WARPS_PER_CTA;
  constexpr int ROWS_PER_WARP = 1;

  int64_t const num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  int64_t const num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);

  using F = void(
      float const* input,
      float const* bias,
      bool const* finished,
      float* output,
      int64_t const num_rows,
      int32_t* indices,
      int64_t num_experts,
      int64_t num_expert_group,
      int64_t topk_group,
      int64_t topk,
      int64_t num_fused_shared_experts,
      double routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output);

  F* f = nullptr;

  if constexpr (IS_POW2) {
    constexpr auto div_ceil = [](auto x, auto y) { return (x + y - 1) / y; };
    switch (num_experts) {
      case 64: {
        constexpr int VPT = div_ceil(64, WARP_SIZE * 4) * 4;
        f = moe_fused_gate_vectorized_pow2<VPT, WARPS_PER_TB, BYTES_PER_LDG>;
        break;
      }
      case 128: {
        constexpr int VPT = div_ceil(128, WARP_SIZE * 4) * 4;
        f = moe_fused_gate_vectorized_pow2<VPT, WARPS_PER_TB, BYTES_PER_LDG>;
        break;
      }
      case 256: {
        constexpr int VPT = div_ceil(256, WARP_SIZE * 4) * 4;
        f = moe_fused_gate_vectorized_pow2<VPT, WARPS_PER_TB, BYTES_PER_LDG>;
        break;
      }
      default:
        TORCH_CHECK(false, "Unsupported num_experts for BYTES_PER_LDG=16: ", num_experts);
    }
  } else {
    // GLM NOTE: num_expert_group is 1
    switch (num_experts) {
      case 96: {
        constexpr int VPT = 96;
        f = moe_fused_gate_vectorized_m32<VPT, WARPS_PER_TB, BYTES_PER_LDG>;
        break;
      }
      case 160: {
        constexpr int VPT = 160;
        f = moe_fused_gate_vectorized_m32<VPT, WARPS_PER_TB, BYTES_PER_LDG>;
        break;
      }
      default:
        TORCH_CHECK(false, "Unsupported num_experts for BYTES_PER_LDG=16: ", num_experts);
    }
  }
  f<<<num_blocks, block_dim, 0, stream>>>(
      input,
      bias,
      finished,
      output,
      num_rows,
      indices,
      num_experts,
      num_expert_group,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output);
}
