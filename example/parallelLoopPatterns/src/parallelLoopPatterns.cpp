/* Copyright 2023 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>
#include <typeinfo>

//! Functor to process the given index in the user data domain.
//! This example demonstrates patterns for data-parallel loops, so the function itself is not a focus.
//! So we simply return some index-dependant value.
//!
//! Note that generally here we operate in the data domain, not in a thread index domain.
//! Some thread will eventually be mapped to calculate this data domain point.
//! This mapping and work distribution is defined by a combination of a kernel and its launch configuration.
//! However, this function is generally independent of that.
//! That is an important distinction that we underline with how the kernels in this examples are written.
//!
//! \param idx The index in the data domain to operate on.
ALPAKA_FN_HOST_ACC float process(uint32_t idx)
{
    return static_cast<float>(idx + 3);
}

//! Test if the given buffer has a correct result.
//!
//! \tparam TQueue The queue type for work to be submitted to.
//! \tparam TBufAcc The device buffer type.
//! \param queue The queue object for work to be submitted to.
//! \param bufAcc The device buffer.
template<typename TQueue, typename TBufAcc>
void testResult(TQueue& queue, TBufAcc& bufAcc)
{
    // Wait for kernel to finish
    alpaka::wait(queue);
    // Copy results to host
    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto bufHost = alpaka::allocBuf<float, uint32_t>(devHost, n);
    alpaka::memcpy(queue, bufHost, bufAcc);
    // Reset values of device buffer
    auto const byte(static_cast<uint8_t>(0u));
    alpaka::memset(queue, bufAcc, byte);
    // Test that all elements were processed
    auto const* result = alpaka::getPtrNative(bufHost);
    bool testPassed = true;
    for(uint32_t i = 0u; i < n; i++)
        testPassed = testPassed && (std::abs(result[i] - process(i)) < 1e-3);
    std::cout << (testPassed ? "Test passed.\n" : "Test failed.\n");
}

//! Helper type to set alpaka kernel launch configuration
using WorkDiv = alpaka::WorkDivMembers<alpaka::DimInt<1u>, uint32_t>;

//! A naive CUDA style kernel processing a single element per thread.
struct NaiveCudaStyleKernel
{
    //! The kernel entry point.
    //!
    //! The work is distributed so that each thread processes a single element.
    //! Global thread indices are identity-mapped to data domain indices.
    //! Global thread indices are identity-mapped to output buffer indices.
    //!
    //! This kernel must be called with a 1d work division having overall >= n threads.
    //! Otherwise, some elements won't be processed.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param result The result array.
    //! \param n The number of elements.
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
    {
        auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        // Cuf off threads that have nothing to do
        if(globalThreadIdx < n)
        {
            // Identity-map other threads to data domain and memory indices
            auto const dataDomainIdx = globalThreadIdx;
            auto const memoryIdx = globalThreadIdx;
            result[memoryIdx] = process(dataDomainIdx);
        }
    }
};

//! Data-parallel processing in naive CUDA style:
//! fixed number of threads per block, number of blocks scales with problem size, fixed work per thread.
//! This strategy should not be used for alpaka accelerators with sequential execution of blocks (e.g.
//! AccCpuOmp2Threads).
//!
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TDev The device of the queue.
//! \tparam TQueue The queue type for work to be submitted to.
//! \tparam TBufAcc The device buffer type.
//! \param queue The queue object for work to be submitted to.
//! \param bufAcc The device buffer.
template<typename TAcc, typename TDev, typename TQueue, typename TBufAcc>
void naiveCudaStyle(TDev& dev, TQueue& queue, TBufAcc& bufAcc)
{
    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const deviceProperties = alpaka::getAccDevProps<TAcc>(dev);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    // With this approach, one normally has a fixed number of threads per block
    // and number of blocks scales with the problem size.
    // We have to use upper integer part in case n is not a multiple of threadsPerBlock.
    // alpaka element layer is not used in this pattern.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    auto const elementsPerThread = 1u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "\nNaive CUDA style processing - each thread processes one data point:\n";
    std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
              << "alpaka element layer not used\n";
    alpaka::exec<TAcc>(queue, workDiv, NaiveCudaStyleKernel{}, alpaka::getPtrNative(bufAcc), n);
    testResult(queue, bufAcc);
}

//! A standard CUDA style grid strided loop kernel.
struct GridStridedLoopKernel
{
    //! The kernel entry point.
    //!
    //! Data elements are distributed between threads using a grid-strided loop.
    //! Each thread processes elements that are global-number-of-threads apart from one another.
    //! The starting offset depends on global thread index.
    //! For G = global thread index, T = global number of threads, N = number of data elements,
    //! the mapping of thread index G to a set of processed data elements {D} is
    //! G -> {D = G + i * T, i = 0, 1, ..., ceil(N / T) - 1 | D < N}.
    //! The same mapping is used for output buffer indices.
    //!
    //! This kernel can be run with any 1d work division.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param result The result array.
    //! \param n The number of elements.
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
    {
        auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        for(uint32_t dataDomainIdx = globalThreadIdx; dataDomainIdx < n; dataDomainIdx += globalThreadExtent)
        {
            auto const memoryIdx = dataDomainIdx;
            result[memoryIdx] = process(dataDomainIdx);
        }
    }
};

//! Data-parallel processing in standard CUDA style:
//! fixed number of threads and blocks, work per thread scales with problem size.
//! This strategy can be used with any alpaka accelerator.
//!
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TDev The device of the queue.
//! \tparam TQueue The queue type for work to be submitted to.
//! \tparam TBufAcc The device buffer type.
//! \param queue The queue object for work to be submitted to.
//! \param bufAcc The device buffer.
template<typename TAcc, typename TDev, typename TQueue, typename TBufAcc>
void gridStridedLoop(TDev& dev, TQueue& queue, TBufAcc& bufAcc)
{
    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const deviceProperties = alpaka::getAccDevProps<TAcc>(dev);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    // With this approach, one normally has a fixed number of threads per block
    // and fixed number of blocks tied to hardware parameters.
    // alpaka element layer is not used in this pattern.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = deviceProperties.m_multiProcessorCount;
    auto const elementsPerThread = 1u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "\nGrid strided loop processing - fixed number of threads and blocks:\n";
    std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
              << "alpaka element layer not used\n";
    alpaka::exec<TAcc>(queue, workDiv, GridStridedLoopKernel{}, alpaka::getPtrNative(bufAcc), n);
    testResult(queue, bufAcc);
}

//! A chunked grid strided loop kernel with each thread processing a chunk of consecutive elements.
struct ChunkedGridStridedLoopKernel
{
    //! The kernel entry point.
    //!
    //! Data chunks are distributed between threads in a grid-strided fashion.
    //! Data elements in the same chunk are consecutive and processed by the same thread.
    //! Each thread processes data elements that are global-number-of-threads * number-of-alpaka-elements
    //! apart from one another.
    //! The starting offset depends on global thread index and number of alpaka elements.
    //! For G = global thread index, T = global number of threads, N = number of data elements,
    //! E = number of alpaka elements, the mapping of thread index G to a set of processed data elements {D} is
    //! G -> {D = G * E + i * T * E + j, i = 0, 1, ..., ceil(N / (T * E)) - 1, j = 0, 1, ..., E - 1 | D < N}.
    //! The same mapping is used for output buffer indices.
    //!
    //! This kernel can be run with any 1d work division.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param result The result array.
    //! \param n The number of elements.
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
    {
        auto const numElements(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        // Additionally could split the loop into peeled and remainder
        for(uint32_t chunkStart = globalThreadIdx * numElements; chunkStart < n;
            chunkStart += globalThreadExtent * numElements)
        {
            // When applicable, this loop can be done in vector fashion
            for(uint32_t dataDomainIdx = chunkStart; (dataDomainIdx < chunkStart + numElements) && (dataDomainIdx < n);
                dataDomainIdx++)
            {
                auto const memoryIdx = dataDomainIdx;
                result[memoryIdx] = process(dataDomainIdx);
            }
        }
    }
};

//! Chunked data-parallel processing in grid strided style:
//! fixed number of threads and blocks, fixed number of elements for chunking, work per thread scales with problem
//! size.
//! This strategy can be used with any alpaka accelerator.
//!
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TDev The device of the queue.
//! \tparam TQueue The queue type for work to be submitted to.
//! \tparam TBufAcc The device buffer type.
//! \param queue The queue object for work to be submitted to.
//! \param bufAcc The device buffer.
template<typename TAcc, typename TDev, typename TQueue, typename TBufAcc>
void chunkedGridStridedLoop(TDev& dev, TQueue& queue, TBufAcc& bufAcc)
{
    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const deviceProperties = alpaka::getAccDevProps<TAcc>(dev);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    // With this approach, one normally has a fixed number of threads per block
    // and fixed number of blocks tied to hardware parameters.
    // Fixed sized alpaka element layer defines chunk size.
    // With 1 element per thread this pattern is same as grid strided loop.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = deviceProperties.m_multiProcessorCount;
    auto const elementsPerThread = 8u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "\nChunked grid strided loop processing - fixed number of threads and blocks:\n";
    std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
              << elementsPerThread << " alpaka elements per thread\n";
    alpaka::exec<TAcc>(queue, workDiv, ChunkedGridStridedLoopKernel{}, alpaka::getPtrNative(bufAcc), n);
    testResult(queue, bufAcc);
}

//! A naive OpenMP style kernel mimicking a pragma omp parallel for loop with no chunk specified.
struct NaiveOpenMPStyleKernel
{
    //! The kernel entry point.
    //!
    //! The work is distributed so that each thread processes a single consecutive range of elements.
    //! The starting offset depends on global thread index and number of data elements.
    //! For G = global thread index, T = global number of threads, N = number of data elements,
    //! the mapping of thread index G to a set of processed data elements {D} is
    //! G -> {D = G * ceil(N / T) + i, i = 0, 1, ..., ceil(N / T) - 1 | D < N}.
    //! The same mapping is used for output buffer indices.
    //!
    //! This kernel can be run with any 1d work division.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param result The result array.
    //! \param n The number of elements.
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
    {
        auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const processPerThread = (n + globalThreadExtent - 1) / globalThreadExtent;
        for(uint32_t dataDomainIdx = globalThreadIdx * processPerThread;
            (dataDomainIdx < (globalThreadIdx + 1) * processPerThread) && (dataDomainIdx < n);
            dataDomainIdx++)
        {
            auto const memoryIdx = dataDomainIdx;
            result[memoryIdx] = process(dataDomainIdx);
        }
    }
};

//! Data-parallel processing in naive OpenMP style:
//! fixed number of threads and blocks, work per thread scales with problem size.
//! This strategy can in principle be used with any alpaka accelerator, but numCores must be adjusted.
//!
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TDev The device of the queue.
//! \tparam TQueue The queue type for work to be submitted to.
//! \tparam TBufAcc The device buffer type.
//! \param queue The queue object for work to be submitted to.
//! \param bufAcc The device buffer.
template<typename TAcc, typename TDev, typename TQueue, typename TBufAcc>
void naiveOpenMPStyle(TDev& dev, TQueue& queue, TBufAcc& bufAcc)
{
    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const deviceProperties = alpaka::getAccDevProps<TAcc>(dev);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];
    auto const numCores = std::max(std::thread::hardware_concurrency(), 1u);

    // With this approach, one normally has a fixed number of threads per block
    // and number of blocks scales with the problem size.
    // alpaka element layer is not used in this pattern.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = numCores;
    auto const elementsPerThread = 1u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "\nNaive OpenMP style processing - each thread processes a single consecutive range of elements:\n";
    std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
              << "alpaka element layer not used\n";
    alpaka::exec<TAcc>(queue, workDiv, NaiveOpenMPStyleKernel{}, alpaka::getPtrNative(bufAcc), n);
    testResult(queue, bufAcc);
}

//! A SIMD OpenMP style kernel mimicking a pragma omp parallel for simd loop with no chunk specified.
struct OpenMPSimdStyleKernel
{
    //! The kernel entry point.
    //!
    //! The work is distributed so that each thread processes a single consective range of elements.
    //! For G = global thread index, T = global number of threads, N = number of data elements,
    //! E = number of alpaka elements, the mapping of thread index G to a set of processed data elements {D} is
    //! G -> {D = G * E * ceil(ceil(N / T) / E) + i, i = 0, 1, ..., E * ceil(ceil(N / T) / E) - 1 | D < N}.
    //! The same mapping is used for output buffer indices.
    //!
    //! This kernel can be run with any 1d work division.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param result The result array.
    //! \param n The number of elements.
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float* result, uint32_t n) const
    {
        auto const numElements(alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        auto const globalThreadExtent(alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        auto const globalThreadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        // This is the number for naive OpenMP style
        auto const naiveProcessPerThread = (n + globalThreadExtent - 1) / globalThreadExtent;
        // Round up to multiple of numElements
        auto const processPerThread = numElements * ((naiveProcessPerThread + numElements - 1) / numElements);
        // Additionally could split the loop into peeled and remainder
        for(uint32_t chunkStart = globalThreadIdx * processPerThread;
            chunkStart < (globalThreadIdx + 1) * processPerThread && (chunkStart < n);
            chunkStart += numElements)
        {
            // When applicable, this loop can be done in vector fashion.
            // Potentially compiler-specific vectorization pragmas can be added here.
            for(uint32_t dataDomainIdx = chunkStart; (dataDomainIdx < chunkStart + numElements) && (dataDomainIdx < n);
                dataDomainIdx++)
            {
                auto const memoryIdx = dataDomainIdx;
                result[memoryIdx] = process(dataDomainIdx);
            }
        }
    }
};

//! Data-parallel processing in SIMD OpenMP style:
//! fixed number of threads and blocks, fixed number of elements for SIMD, work per thread scales with problem
//! size.
//! This strategy can in principle be used with any alpaka accelerator, but numCores must be adjusted.
//!
//! \tparam TAcc The accelerator environment to be executed on.
//! \tparam TDev The device of the queue.
//! \tparam TQueue The queue type for work to be submitted to.
//! \tparam TBufAcc The device buffer type.
//! \param queue The queue object for work to be submitted to.
//! \param bufAcc The device buffer.
template<typename TAcc, typename TDev, typename TQueue, typename TBufAcc>
void openMPSimdStyle(TDev& dev, TQueue& queue, TBufAcc& bufAcc)
{
    auto const n = alpaka::getExtentProduct(bufAcc);
    auto const deviceProperties = alpaka::getAccDevProps<TAcc>(dev);
    auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];
    auto const numCores = 16u; // should be taken from hardware properties, alpaka currently does not expose it

    // With this approach, one normally has a fixed number of threads per block
    // and number of blocks scales with the problem size.
    // Fixed sized alpaka element layer defines SIMD size.
    // With 1 element per thread this pattern is same as naive OpenMP style.
    auto const threadsPerBlock = maxThreadsPerBlock;
    auto const blocksPerGrid = numCores;
    auto const elementsPerThread = 4u;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};
    std::cout << "\nOpenMP SIMD style processing - each thread processes a single consecutive range of elements:\n";
    std::cout << "   " << blocksPerGrid << " blocks, " << threadsPerBlock << " threads per block, "
              << elementsPerThread << " alpaka elements per thread\n";
    alpaka::exec<TAcc>(queue, workDiv, OpenMPSimdStyleKernel{}, alpaka::getPtrNative(bufAcc), n);
    testResult(queue, bufAcc);
}

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else

    // Define the index domain, this example is only for 1d
    using Dim = alpaka::DimInt<1u>;

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    // using Acc = alpaka::AccCpuSerial<Dim, uint32_t>;
    using Acc = alpaka::ExampleDefaultAcc<Dim, uint32_t>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Select a device and create queue for it
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>(devAcc);

    // Define the problem size = buffer size and allocate memory
    uint32_t const bufferSize = 1317u;
    auto bufAcc = alpaka::allocBuf<float, uint32_t>(devAcc, bufferSize);

    // Call different kernel versions
    naiveCudaStyle<Acc>(devAcc, queue, bufAcc);
    gridStridedLoop<Acc>(devAcc, queue, bufAcc);
    chunkedGridStridedLoop<Acc>(devAcc, queue, bufAcc);
    naiveOpenMPStyle<Acc>(devAcc, queue, bufAcc);
    openMPSimdStyle<Acc>(devAcc, queue, bufAcc);

#endif
}
