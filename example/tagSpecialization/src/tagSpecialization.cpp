/* Copyright 2023 Simeon Ehrig, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <iostream>

//! Different functions and kernels to illustrate specialization for a particular accelerator with tags without pre
//! processors guards.
//! The usage of tags for specialization requires the following conditions:
//!   - the code inside a specialized functions needs to be compileable independent of the enabled accelerators -> e.g.
//!     built-in function like cuda functions are not allowed
//!   - entry kernels, which are executed with alpaka::exec cannot be specialized with tags

//! Function specialization via template specialization
template<typename TAcc>
std::string host_function_ver1()
{
    return "generic host function v1";
}

template<>
std::string host_function_ver1<alpaka::TagGpuCudaRt>()
{
    return "CUDA host function v1";
}

//! Function specialization via overloading
template<typename TTag>
std::string host_function_ver2(TTag)
{
    return "generic host function v2";
}

std::string host_function_ver2(alpaka::TagGpuCudaRt)
{
    return "CUDA host function v2";
}

//! Kernel specialization via SFINAE
//! Allows to specialize a function for more than one Acc
//! Can be also used for normal functions
template<typename TAcc, typename = void>
struct specialized_Kernel
{
    template<typename TData>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC TData operator()(TAcc const& acc, TData const input)
    {
        printf("generic kernel\n");
        return alpaka::math::min(acc, 0, input);
    }
};

template<typename TAcc>
struct specialized_Kernel<
    TAcc,
    std::enable_if_t<alpaka::accMatchesTags<TAcc, alpaka::TagCpuOmp2Blocks, alpaka::TagCpuOmp2Threads>>>
{
    template<typename TData>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC TData operator()(TAcc const& acc, TData const input)
    {
        printf("OpenMP kernel\n");
        return alpaka::math::min(acc, 1, input);
    }
};

struct WrapperKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const noexcept -> void
    {
        int const data = 42;
        printf("value of the kernel: %i\n", specialized_Kernel<TAcc>{}(acc, data));
    }
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
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp2Blocks
    // - AccCpuTbbBlocks
    // - AccCpuSerial
    //
    // For simplicity this examples always uses 1 dimensional indexing, and index type size_t
    using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, std::size_t>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Call the specialized functions
    std::cout << host_function_ver1<alpaka::AccToTag<Acc>>() << std::endl;
    std::cout << host_function_ver2(alpaka::AccToTag<Acc>{}) << std::endl;

    // Defines the synchronization behavior of a queue
    using QueueProperty = alpaka::Blocking;
    using Queue = alpaka::Queue<Acc, QueueProperty>;
    // Select a device
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    Queue queue(devAcc);

    // Define the work division
    // Run only one thread independent of the acc
    alpaka::WorkDivMembers<alpaka::DimInt<1>, std::size_t> workDiv{
        static_cast<std::size_t>(1),
        static_cast<std::size_t>(1),
        static_cast<std::size_t>(1)};

    // Run the wrapper kernel, which calls the actual specialized
    // Specializing the entry kernel with tags is not possible. Therefore pre processor guards are required, see
    // kernelSpecialization example.
    alpaka::exec<Acc>(queue, workDiv, WrapperKernel{});
    alpaka::wait(queue);
    return EXIT_SUCCESS;
#endif
}
