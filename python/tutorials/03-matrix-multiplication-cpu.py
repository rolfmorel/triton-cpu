"""
Matrix Multiplication
=====================
In this tutorial, you will write a very short high-performance FP32 matrix multiplication kernel.

You will specifically learn about:

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.

"""

# %%
# Motivations
# -----------
#
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # Do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # Do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

# %%
# Compute Kernel
# --------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
# of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetic.
#
# Pointer Arithmetic
# ~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given
# by :code:`&X[i, j] = X + i*stride_xi + j*stride_xj`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as the following
# code. Also note that we need an extra modulo to handle the case where :code:`M` is not a multiple of
# :code:`BLOCK_SIZE_M` or :code:`N` is not a multiple of :code:`BLOCK_SIZE_N`, in which case we can pad the data with
# some useless values, which will not contribute to the results. For the :code:`K` dimension, we will handle that later
# using masking load semantics.
#
#  .. code-block:: python
#
#    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#    offs_k = tl.arange(0, BLOCK_SIZE_K)
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    a_ptrs += BLOCK_SIZE_K * stride_ak;
#    b_ptrs += BLOCK_SIZE_K * stride_bk;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]`
# block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program, and unfortunately, a
# simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = triton.program_id(0);
#    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M;
#    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N;
#    pid_m = pid / grid_n;
#    pid_n = pid % grid_n;
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    # Program ID
#    pid = tl.program_id(axis=0)
#    # Number of program ids along the M axis
#    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#    # Number of programs ids along the N axis
#    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # Number of programs in group
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # Id of the group this program is in
#    group_id = pid // num_pid_in_group
#    # Row-id of the first program in the group
#    first_pid_m = group_id * GROUP_SIZE_M
#    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *Within groups*, programs are ordered in a column-major order
#    # Row-id of the program in the *launch grid*
#    pid_m = first_pid_m + (pid % group_size_m)
#    # Col-id of the program in the *launch grid*
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# ------------

import torch
import math

import triton
import triton.language as tl

# It depends on CPU cache sizes.
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 8
USE_GPU = False
USE_BLOCK_POINTERS = False
DATA_TYPE = torch.float32
K_DIM_PADDING = False
DYNAMIC_K_BLOCK = False

@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, # arg0
        b_ptr, # arg1
        c_ptr, # arg2
        # Matrix dimensions
        M, # arg3
        N, # arg4
        K, # arg5
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, # arg6
        stride_ak, # arg7
        stride_bk, # arg8
        stride_bn, # arg9
        stride_cm, # arg11
        stride_cn, # arg12
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        USE_BLOCK_POINTERS: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if USE_BLOCK_POINTERS:
        block_offset_m = pid_m * BLOCK_SIZE_M
        block_offset_n = pid_n * BLOCK_SIZE_N

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    if USE_BLOCK_POINTERS:
        a_tile_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(block_offset_m, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0)
        )
        b_tile_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, block_offset_n),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1, 0)
        )
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to matrix C's type after the loop, if C has lower precision type (for example, float16 and bfloat16).
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.

        if USE_BLOCK_POINTERS:
            # TODO: Currently masked load is not supported yet.
            a = tl.load(a_tile_ptr, boundary_check=(0, 1))
            b = tl.load(b_tile_ptr, boundary_check=(0, 1))
        else:
            # TODO: Currently masked load is not supported yet.
            # a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            # b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        # Advance the ptrs to the next K block.
        if USE_BLOCK_POINTERS:
            a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_SIZE_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_SIZE_K, 0])
        else:
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert the accumulator to the output matrix C's type if needed.
    c = accumulator.to(c_ptr.type.element_ty)

    if USE_BLOCK_POINTERS:
        # TODO: masking
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(block_offset_m, block_offset_n),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(1, 0)
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))
    else:
        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

        # TODO: Currently masked load is not supported yet.
        # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        # tl.store(c_ptrs, c, mask=c_mask)
        tl.store(c_ptrs, c)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, num_threads=0):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape

    k_block = BLOCK_SIZE_K
    if DYNAMIC_K_BLOCK:
        # Currently, the maximum dynamic block size is capped somewhat arbitrarily.
        # Ideally, tradeoffs between amount of padding, block size, and associated costs
        # should be considered.
        k_block = min(triton.next_power_of_2(K), 1024)

    if K_DIM_PADDING or DYNAMIC_K_BLOCK:
        padding_size = (math.ceil(K / k_block) * k_block) - K
        if padding_size != 0:
            a = torch.nn.functional.pad(a, (0, padding_size, 0, 0), mode='constant', value=0)
            b = torch.nn.functional.pad(b, (0, 0, 0, padding_size), mode='constant', value=0)
            K = a.shape[1]

    #TODO: Currently masked load is not supported yet.
    assert (M % BLOCK_SIZE_M == 0) and (N % BLOCK_SIZE_N == 0) and (
        K % k_block == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    if c is None:
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert c.shape == (M, N), "Incompatible dimensions"
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=k_block,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
        num_threads=num_threads,  #
        USE_BLOCK_POINTERS=USE_BLOCK_POINTERS,  #
    )
    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation.

torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

a = torch.randn((512, 512), device='cpu', dtype=DATA_TYPE)
b = torch.randn((512, 512), device='cpu', dtype=DATA_TYPE)
triton_output = matmul(a, b, None)
torch_output = torch.matmul(a, b)
print(f"triton_cpu_output_with_{a.dtype}_inputs={triton_output}")
print(f"torch_cpu_output_with_{a.dtype}_inputs={torch_output}")
rtol = 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU and TorchCPU match")
elif DATA_TYPE == torch.bfloat16 and torch.allclose(triton_output, torch_output, atol=2e-0, rtol=rtol):
    print("⚠️ TritonCPU and TorchCPU rounding errors, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
else:
    print("❌ TritonCPU and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of Pytorch. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

# LINE_VALS = [
#     'triton-cpu-single', 'triton-cpu', 'triton-cpu-single-v2', 'triton-cpu-v2', 'torch-cpu-native', 'torch-cpu-compile']
# LINE_NAMES = ['TritonCPU 1', 'TritonCPU', 'TritonCPU 1-v2', 'TritonCPU-v2', 'TorchCPU (native)', 'TorchCPU (compile)']
# LINE_STYLES = [('blue', '--'), ('blue', '-'), ('red', '--'), ('red', '-'), ('green', '--'), ('green', '-')]

# Disabled v2 benchmarking.
# v2 lowering effectively fails for tiles larger than 16 and throws errors on bf16 data type.
LINE_VALS = [
    'triton-cpu-single', 'triton-cpu', 'torch-cpu-native', 'torch-cpu-compile']
LINE_NAMES = ['TritonCPU 1', 'TritonCPU', 'TorchCPU (native)', 'TorchCPU (compile)']
LINE_STYLES = [('blue', '--'), ('blue', '-'), ('green', '--'), ('green', '-')]

if USE_GPU and triton.runtime.driver.get_active_gpus():
    triton.runtime.driver.set_active_to_gpu()
    a = a.to('cuda')
    b = b.to('cuda')
    triton_output = matmul(a, b, None)
    torch_output = torch.matmul(a, b)
    print(f"triton_gpu_output_with_{a.dtype}_inputs={triton_output}")
    print(f"torch_gpu_output_with_{a.dtype}_inputs={torch_output}")
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ TritonGPU and TorchGPU match")
    else:
        print("❌ TritonGPU and TorchGPU differ, the maximum difference is "
              f'{torch.max(torch.abs(triton_output - torch_output))}')

    LINE_VALS += ['triton-gpu', 'torch-gpu']
    LINE_NAMES += ['TritonGPU', 'TorchGPU']
    LINE_STYLES += [('yellow', '-'), ('red', '-')]

# %%
# Seems like we're good to go!

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.

STR_TYPE = str(DATA_TYPE).rsplit('.')[-1]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 21)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GFLOPS',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'matmul-performance-{STR_TYPE} (BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M})',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(M, N, K, provider):

    device = 'cpu' if 'cpu' in provider else 'cuda'
    a = torch.randn((M, K), device=device, dtype=DATA_TYPE)
    b = torch.randn((K, N), device=device, dtype=DATA_TYPE)

    if device == 'cpu':
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        triton.runtime.driver.set_active_to_cpu()
    else:
        c = None
        triton.runtime.driver.set_active_to_gpu()

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-gpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == 'triton-gpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, None), quantiles=quantiles)
    elif provider == 'torch-cpu-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b, out=c), quantiles=quantiles)
    elif provider == 'torch-cpu-compile':
        compiled = torch.compile(torch.matmul)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled(a, b, out=c), quantiles=quantiles)
    elif provider == 'triton-cpu-single':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c, num_threads=1), quantiles=quantiles)
    elif provider == 'triton-cpu':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-9 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
