/* Copyright 2023 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <cstdint>
#include <iostream>

//! Complex numbers demonstration kernel
struct ComplexKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc) const -> void
    {
        // alpaka::Complex<T> supports the same methods as std::complex<T>, they are also useable inside kernels
        auto x = alpaka::Complex<float>(0.1f, 0.2f);
        float const real = x.real();
        auto y = alpaka::Complex<float>(0.3f, 0.4f);

        // Operators are also the same
        x *= 2.0f;
        alpaka::Complex<float> z = x + y;

        // In-kernel math functions are accessed via alpaka wrappers, the same way as for real numbers
        float zAbs = alpaka::math::abs(acc, z);
    }
};

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    using Idx = std::size_t;

    // Define the accelerator
    using Acc = alpaka::TagToAcc<TAccTag, alpaka::DimInt<1>, Idx>;
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
    Idx const elementsPerGrid = 1u;
    Idx const elementsPerThread = 1u;

    ComplexKernel complexKernel;

    alpaka::KernelCfg<Acc> const kernelCfg = {elementsPerGrid, elementsPerThread};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, complexKernel);

    // Run the kernel
    alpaka::exec<Acc>(queue, workDiv, complexKernel);
    alpaka::wait(queue);

    // Usage of alpaka::Complex<T> on the host side is the same as inside kernels, except math functions are not
    // supported
    auto x = alpaka::Complex<float>(0.1f, 0.2f);
    float const real = x.real();
    auto y = alpaka::Complex<float>(0.3f, 0.4f);
    x *= 2.0f;
    alpaka::Complex<float> z = x + y;

    return EXIT_SUCCESS;
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
