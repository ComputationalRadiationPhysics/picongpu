[:arrow_up: Up](../Mapping.md)

CUDA GPUs
=========

Mapping the abstraction to GPUs supporting *CUDA* is straightforward because the hierarchy levels are identical up to the element level.
So blocks of warps of threads will be mapped directly to their *CUDA* equivalent.

The element level is supported through an additional run-time variable containing the extent of elements per thread.
This variable can be accessed by all threads and should optimally be placed in constant device memory for fast access.

Porting CUDA to *alpaka*
------------------------

Nearly all CUDA functionality can be directly mapped to alpaka function calls.
A major difference is that CUDA requires the block and grid sizes to be given in (x, y, z) order. Alpaka uses the mathematical C/C++ array indexing scheme [z][y][x]. In both cases x is the innermost / fast running index.

Furthermore alpaka does not require the indices and extents to be 3-dimensional.
The accelerators are templatized on and support arbitrary dimensionality.
NOTE: Currently the CUDA implementation is restricted to a maximum of 3 dimensions!

NOTE: You have to be careful when mixing alpaka and non alpaka CUDA code. The CUDA-accelerator back-end can change the current CUDA device and will NOT set the device back to the one prior to the invocation of the alpaka function.


### Programming Interface

*Function Attributes*

|CUDA|alpaka|
|---|---|
|\_\_host\_\_|ALPAKA_FN_HOST|
|\_\_device\_\_|ALPAKA_FN_ACC*|
|\_\_global\_\_|ALPAKA_FN_ACC*|
|\_\_host\_\_ \_\_device\_\_|ALPAKA_FN_HOST_ACC|

\* You can not call CUDA only methods except when ALPAKA_ACC_GPU_CUDA_ONLY_MODE is enabled.

*Memory*

|CUDA|alpaka|
|---|---|
|\_\_shared\_\_|[alpaka::block::shared::st::allocVar<std::uint32_t, \_\_COUNTER\_\_>(acc)](../../../../../test/unit/block/shared/src/BlockSharedMemSt.cpp#L69)|
|\_\_constant\_\_|[ALPAKA_STATIC_ACC_MEM_CONSTANT](../../../../../test/unit/mem/view/src/ViewStaticAccMem.cpp#L58-L63)|
|\_\_device\_\_|[ALPAKA_STATIC_ACC_MEM_GLOBAL](../../../../../test/unit/mem/view/src/ViewStaticAccMem.cpp#L164-L169)|

*Index / Work Division*

|CUDA|alpaka|
|---|---|
|threadIdx|alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)|
|blockIdx|alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)|
|blockDim|alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)|
|gridDim|alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)|

*Types*

|CUDA|alpaka|
|---|---|
|dim3|[alpaka::vec::Vec< TDim, TVal >](../../../../../test/unit/vec/src/VecTest.cpp#L43-L45)|


### CUDA Runtime API

The following tables list the functions available in the [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules) and their equivalent alpaka functions:

*Device Management*

|CUDA|alpaka|
|---|---|
|cudaChooseDevice|-|
|cudaDeviceGetAttribute|-|
|cudaDeviceGetByPCIBusId|-|
|cudaDeviceGetCacheConfig|-|
|cudaDeviceGetLimit|-|
|cudaDeviceGetP2PAttribute|-|
|cudaDeviceGetPCIBusId|-|
|cudaDeviceGetSharedMemConfig|-|
|cudaDeviceGetQueuePriorityRange|-|
|cudaDeviceReset|alpaka::dev::reset(device)|
|cudaDeviceSetCacheConfig|-|
|cudaDeviceSetLimit|-|
|cudaDeviceSetSharedMemConfig|-|
|cudaDeviceSynchronize|void alpaka::wait::wait(device)|
|cudaGetDevice|n/a (no current device)|
|cudaGetDeviceCount|std::size_t alpaka::pltf::getDevCount< TPltf >()|
|cudaGetDeviceFlags|-|
|cudaGetDeviceProperties|alpaka::acc::getAccDevProps(dev) *NOTE: Only some properties available*|
|cudaIpcCloseMemHandle|-|
|cudaIpcGetEventHandle|-|
|cudaIpcGetMemHandle|-|
|cudaIpcOpenEventHandle|-|
|cudaIpcOpenMemHandle|-|
|cudaSetDevice|n/a (no current device)|
|cudaSetDeviceFlags|-|
|cudaSetValidDevices|-|

*Error Handling*

|CUDA|alpaka|
|---|---|
|cudaGetErrorName|n/a (handled internally, available in exception message)|
|cudaGetErrorString|n/a (handled internally, available in exception message)|
|cudaGetLastError|n/a (handled internally)|
|cudaPeekAtLastError|n/a (handled internally)|

*Queue Management*

|CUDA|alpaka|
|---|---|
|cudaStreamAddCallback|alpaka::queue::enqueue(queue, \[\](){do_something();})|
|cudaStreamAttachMemAsync|-|
|cudaStreamCreate|<ul><li>queue = alpaka::queue::QueueCudaRtNonBlocking(device);</li><li>queue = alpaka::queue::QueueCudaRtBlocking(device);</li></ul>|
|cudaStreamCreateWithFlags|see cudaStreamCreate (cudaStreamNonBlocking hard coded)|
|cudaStreamCreateWithPriority|-|
|cudaStreamDestroy|n/a (Destructor)|
|cudaStreamGetFlags|-|
|cudaStreamGetPriority|-|
|cudaStreamQuery|bool alpaka::queue::empty(queue)|
|cudaStreamSynchronize|void alpaka::wait::wait(queue)|
|cudaStreamWaitEvent|void alpaka::wait::wait(queue, event)|

*Event Management*

|CUDA|alpaka|
|---|---|
|cudaEventCreate|alpaka::event::Event< TQueue > event(dev);|
|cudaEventCreateWithFlags|-|
|cudaEventDestroy|n/a (Destructor)|
|cudaEventElapsedTime|-|
|cudaEventQuery|bool alpaka::event::test(event)|
|cudaEventRecord|void alpaka::queue::enqueue(queue, event)|
|cudaEventSynchronize|void alpaka::wait::wait(event)|

*Memory Management*

|CUDA|alpaka|
|---|---|
|cudaArrayGetInfo|-|
|cudaFree|n/a (automatic memory management with reference counted memory handles)|
|cudaFreeArray|-|
|cudaFreeHost|n/a|
|cudaFreeMipmappedArray|-|
|cudaGetMipmappedArrayLevel|-|
|cudaGetSymbolAddress|-|
|cudaGetSymbolSize|-|
|cudaHostAlloc|n/a, the existing buffer can be pinned using alpaka::mem::buf::prepareForAsyncCopy(memBuf)|
|cudaHostGetDevicePointer|-|
|cudaHostGetFlags|-|
|cudaHostRegister|-|
|cudaHostUnregister|-|
|cudaMalloc|alpaka::mem::buf::alloc<TElement>(device, extents1D)|
|cudaMalloc3D|alpaka::mem::buf::alloc<TElement>(device, extents3D)|
|cudaMalloc3DArray|-|
|cudaMallocArray|-|
|cudaMallocHost|alpaka::mem::buf::alloc<TElement>(device, extents) *1D, 2D, 3D suppoorted!*|
|cudaMallocManaged|-|
|cudaMallocMipmappedArray|-|
|cudaMallocPitch|alpaka::mem::alloc<TElement>(device, extents2D)|
|cudaMemAdvise|-|
|cudaMemGetInfo|<ul><li>alpaka::dev::getMemBytes</li><li>alpaka::dev::getFreeMemBytes</li><ul>|
|cudaMemPrefetchAsync|-|
|cudaMemRangeGetAttribute|-|
|cudaMemRangeGetAttributes|-|
|cudaMemcpy|alpaka::mem::view::copy(memBufDst, memBufSrc, extents1D)|
|cudaMemcpy2D|alpaka::mem::view::copy(memBufDst, memBufSrc, extents2D)|
|cudaMemcpy2DArrayToArray|-|
|cudaMemcpy2DAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents2D, queue)|
|cudaMemcpy2DFromArray|-|
|cudaMemcpy2DFromArrayAsync|-|
|cudaMemcpy2DToArray|-|
|cudaMemcpy2DToArrayAsync|-|
|cudaMemcpy3D|alpaka::mem::view::copy(memBufDst, memBufSrc, extents3D)|
|cudaMemcpy3DAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents3D, queue)|
|cudaMemcpy3DPeer|alpaka::mem::view::copy(memBufDst, memBufSrc, extents3D)|
|cudaMemcpy3DPeerAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents3D, queue)|
|cudaMemcpyArrayToArray|-|
|cudaMemcpyAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents1D, queue)|
|cudaMemcpyFromArray|-|
|cudaMemcpyFromArrayAsync|-|
|cudaMemcpyFromSymbol|-|
|cudaMemcpyFromSymbolAsync|-|
|cudaMemcpyPeer|alpaka::mem::view::copy(memBufDst, memBufSrc, extents1D)|
|cudaMemcpyPeerAsync|alpaka::mem::view::copy(memBufDst, memBufSrc, extents1D, queue)|
|cudaMemcpyToArray|-|
|cudaMemcpyToArrayAsync|-|
|cudaMemcpyToSymbol|-|
|cudaMemcpyToSymbolAsync|-|
|cudaMemset|alpaka::mem::view::set(memBufDst, byte, extents1D)|
|cudaMemset2D|alpaka::mem::view::set(memBufDst, byte, extents2D)|
|cudaMemset2DAsync|alpaka::mem::view::set(memBufDst, byte, extents2D, queue)|
|cudaMemset3D|alpaka::mem::view::set(memBufDst, byte, extents3D)|
|cudaMemset3DAsync|alpaka::mem::view::set(memBufDst, byte, extents3D, queue)|
|cudaMemsetAsync|alpaka::mem::view::set(memBufDst, byte, extents1D, queue)|
|make_cudaExtent|-|
|make_cudaPitchedPtr|-|
|make_cudaPos|-|
|cudaMemcpyHostToDevice|n/a (direction of copy is determined automatically)|
|cudaMemcpyDeviceToHost|n/a (direction of copy is determined automatically)|

*Execution Control*

|CUDA|alpaka|
|---|---|
|cudaFuncGetAttributes|-|
|cudaFuncSetCacheConfig|-|
|cudaFuncSetSharedMemConfig|-|
|cudaLaunchKernel|<ul><li>alpaka::kernel::exec< TAcc >(queue, workDiv, kernel, params...)</li><li>alpaka::kernel::BlockSharedExternMemSizeBytes< TKernel< TAcc > >::getBlockSharedExternMemSizeBytes<...>(...)</li></ul>|
|cudaSetDoubleForDevice|n/a (alpaka assumes double support)|
|cudaSetDoubleForHost|n/a (alpaka assumes double support)|

*Occupancy*

|CUDA|alpaka|
|---|---|
|cudaOccupancyMaxActiveBlocksPerMultiprocessor|-|
|cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags|-|


*Unified Addressing*

|CUDA|alpaka|
|---|---|
|cudaPointerGetAttributes|-|

*Peer Device Memory Access*

|CUDA|alpaka|
|---|---|
|cudaDeviceCanAccessPeer|-|
|cudaDeviceDisablePeerAccess|-|
|cudaDeviceEnablePeerAccess|automatically done when required|

**OpenGL, Direct3D, VDPAU, EGL, Graphics Interoperability**

*not available*

**Texture/Surface Reference/Object Management**

*not available*

**Version Management**

*not available*
