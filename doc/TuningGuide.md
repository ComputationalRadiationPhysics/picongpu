How to Tune Your *cupla* Kernels
================================

This guide assume that you have [ported your code to cupla](PortingGuide.md).


Add the alpaka Element Level to Your Kernels
============================================

- With `elemDim` you get the number of elements per thread for each dimension.
- There is no `elemIdx` for indexing, elements must be controlled by the user.

**CUDA kernel code**
```C++
// ...
int idx = blockIdx.x * blockDim.x + threadIdx.x;

g_ptr[idx] = g_ptr[idx] + value;
__syncthreads();
g_ptr[idx] = 42;
// ...
```

**CUDA kernel code**
```C++
// ...
// in this example the element layer is equivalent to the thread layer
int idx = blockIdx.x * (blockDim.x * elemDim.x) + threadIdx.x;
for(int i = 0; i < elemDim.x; ++i)
    g_ptr[idx + i] = g_ptr[idx + i] + value;
__syncthreads();
for(int i = 0; i < elemDim.x; ++i)
    g_ptr[idx + i] = 42;
// ...
```

- You can still use the kernel call macro `CUPLA_KERNEL(kernelName)(...)(...)`
  which always sets the number of elements per thread to `dim3(1,1,1)`
- If the kernel runs with `CUPLA_KERNEL` you can change the kernel call to
  `CUPLA_KERNEL_OPTI(kernelName)(...)(...)`
  - `CUPLA_KERNEL_OPTI` has the same parameter signature as `CUPLA_KERNEL` but 
    depending on the selected accelerator the number of elements per thread `dim3(1,1,1)`
    is swapped with the `blockSize` 
  - The preferred x86 accelerator is `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE`
- For the last tuning step you can switch to the kernel call 
  `CUPLA_KERNEL_ELEM(kernelname)(gridSize,blockSize,elemSize,...)(...)` with the full
  explicit control over the sizes of all layers. This assumes that you write
  your own method to calculate the best fitting kernel start parameters for the
  selected accelerator device.
  - alpaka provides macro `define`s for each selected device type like
        - `ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED`
        - `ALPAKA_ACC_GPU_CUDA_ENABLED`
        - `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE`
        - ...
  - Or you can use your own traits and the specification from the cupla
    accelerator (`cupla::Acc` defined in [`types.hpp`](../include/cupla/types.hpp))
    to calculate valid kernel start parameters.

example `CUPLA_KERNEL_ELEM`
```C++
// ...
dim3 gridSize(42,1,1);
dim3 blockSize(256,1,1);
dim3 elemSize(2,1,1);
// extern shared memory and stream is optional
CUPLA_KERNEL_ELEM(fooKernel< 16 >)( gridSize, blockSize, elemSize, 0, 0 )( ptr, 23 );
// ...
```

- To maximize your kernel performance you need to abstract your kernel access pattern
  depending on the currently used accelerator.
  - The memory access pattern abstraction is not part of **cupla** or **alpaka**,
    (nor is it part of CUDA).
  - The Nvidia CUDA accelerator `ALPAKA_ACC_GPU_CUDA_ENABLE` works well with a
    stridden access pattern to avoid e.g, shared memory bank conflicts and to
    allow contiguous global memory access.
  - x86 accelerators `ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE`, `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE`
    and `ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE` benefit from cache optimized
    access patterns where one thread works on a coherent memory line.
