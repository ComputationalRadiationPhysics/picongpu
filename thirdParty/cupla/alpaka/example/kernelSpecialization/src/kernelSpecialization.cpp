/* Copyright 2023 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>

//! Kernel to illustrate specialization for a particular accelerator
//!
//! It has a generic operator() implementation and an overload for the CUDA accelerator.
//! When running the kernel on a CUDA device, the corresponding overload of operator() is called.
//! Otherwise the generic version is called.
//! The same technique can be applied for any function called from inside the kernel,
//! thus allowing specialization of only relevant part of the code.
//! It can be useful for optimization or accessing specific functionality not abstracted by alpaka.
//!
//! This kernel demonstrates the simplest way to achieve the effect by function overloading.
//! Note that it does not perform specialization (in C++ template meaning) of function templates.
//! We use the word "specialization" as it represents a case of having a special version for a particular accelerator.
//! One could apply a similar technique by having an additional class template parametrized by the accelerator type.
//! For such a case, both template specialization and function overloading of the methods can be employed.
struct Kernel
{
    //! Implementation for the general case
    //!
    //! It will be called when no overload is a better match.
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const
    {
        // For simplicity assume 1d thread indexing
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if(globalThreadIdx == 0u)
            printf("Running the general kernel implementation\n");
    }

    //! Simple overload to have a special version for the CUDA accelerator
    //!
    //! We have to guard it with #ifdef as the types of alpaka accelerators are only conditionally available.
    //! Overloading for other accelerators is similar, with another template name instead of AccGpuCudaRt.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<typename TDim, typename TIdx>
    ALPAKA_FN_ACC auto operator()(alpaka::AccGpuCudaRt<TDim, TIdx> const& acc) const
    {
        // This overload is used when the kernel is run on the CUDA accelerator.
        // So inside we can use both alpaka and native CUDA directly.
        // For simplicity assume 1d thread indexing
        auto const globalThreadIdx = blockIdx.x * gridDim.x + threadIdx.x;
        if(globalThreadIdx == 0)
            printf("Running the specialization for the CUDA accelerator\n");
    }
#endif
};

auto main() -> int
{
// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else

    // Define the accelerator
    //
    // It is possible to choose from a set of accelerators:
    // - AccGpuCudaRt
    // - AccGpuHipRt
    // - AccCpuThreads
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    //
    // For simplicity this examples always uses 1 dimensional indexing, and index type size_t
    using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, std::size_t>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Defines the synchronization behavior of a queue
    using QueueProperty = alpaka::Blocking;
    using Queue = alpaka::Queue<Acc, QueueProperty>;

    // Select a device
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    std::size_t const threadsPerGrid = 16u;
    std::size_t const elementsPerThread = 1u;
    auto const workDiv = alpaka::getValidWorkDiv<Acc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    // Run the kernel
    alpaka::exec<Acc>(queue, workDiv, Kernel{});
    alpaka::wait(queue);

    return EXIT_SUCCESS;
#endif
}
