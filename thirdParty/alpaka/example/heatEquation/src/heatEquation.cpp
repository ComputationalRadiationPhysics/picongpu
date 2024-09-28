/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! Solving equation u_t(x, t) = u_xx(x, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x
//!
//! \param uCurrBuf grid values of u for each x and the current value of t:
//!                 u(x, t) | t = t_current
//! \param uNext resulting grid values of u for each x and the next value of t:
//!              u(x, t) | t = t_current + dt
//! \param extent number of grid nodes in x (eq. to numNodesX)
//! \param dx step in x
//! \param dt step in t

struct HeatEquationKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double const* const uCurrBuf,
        double* const uNextBuf,
        uint32_t const extent,
        double const dx,
        double const dt) const -> void
    {
        // Each kernel executes one element
        double const r = dt / (dx * dx);
        int idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if(idx > 0 && idx < extent - 1u)
        {
            uNextBuf[idx] = uCurrBuf[idx] * (1.0 - 2.0 * r) + uCurrBuf[idx - 1] * r + uCurrBuf[idx + 1] * r;
        }
    }
};

//! Exact solution to the test problem
//! u_t(x, t) = u_xx(x, t), x in [0, 1], t in [0, T]
//! u(0, t) = u(1, t) = 0
//! u(x, 0) = sin(pi * x)
//!
//! \param x value of x
//! \param t value of t
auto exactSolution(double const x, double const t) -> double
{
    constexpr double pi = 3.14159265358979323846;
    return std::exp(-pi * pi * t) * std::sin(pi * x);
}

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any available accelerator.
//! Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
//! selected accelerator only. If you use the example as the starting point for your project, you can rename the
//! example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Parameters (a user is supposed to change numNodesX, numTimeSteps)
    uint32_t const numNodesX = 1000;
    uint32_t const numTimeSteps = 10000;
    double const tMax = 0.001;
    // x in [0, 1], t in [0, tMax]
    double const dx = 1.0 / static_cast<double>(numNodesX - 1);
    double const dt = tMax / static_cast<double>(numTimeSteps - 1);

    // Check the stability condition
    double const r = dt / (dx * dx);
    if(r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/dx^2 = " << r << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Set Dim and Idx type
    using Dim = alpaka::DimInt<1u>;
    using Idx = uint32_t;

    // Define the accelerator
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Select specific devices
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Get valid workdiv for the given problem
    uint32_t elemPerThread = 1;
    alpaka::Vec<Dim, Idx> const extent{numNodesX};

    // Select queue
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{devAcc};

    // Initialize host-buffer
    // This buffer holds the calculated values
    auto uNextBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);
    // This buffer will hold the current values (used for the next step)
    auto uCurrBufHost = alpaka::allocBuf<double, Idx>(devHost, extent);

    double* const pCurrHost = std::data(uCurrBufHost);
    double* const pNextHost = std::data(uNextBufHost);

    // Accelerator buffer
    using BufAcc = alpaka::Buf<Acc, double, Dim, Idx>;
    auto uNextBufAcc = BufAcc{alpaka::allocBuf<double, Idx>(devAcc, extent)};
    auto uCurrBufAcc = BufAcc{alpaka::allocBuf<double, Idx>(devAcc, extent)};

    double* pCurrAcc = std::data(uCurrBufAcc);
    double* pNextAcc = std::data(uNextBufAcc);

    // Apply initial conditions for the test problem
    for(uint32_t i = 0; i < numNodesX; i++)
    {
        pCurrHost[i] = exactSolution(i * dx, 0.0);
    }

    HeatEquationKernel heatEqKernel;

    alpaka::KernelCfg<Acc> const kernelCfg = {extent, elemPerThread};

    // Let alpaka calculate good block and grid sizes given our full problem extent
    auto const workDiv
        = alpaka::getValidWorkDiv(kernelCfg, devAcc, heatEqKernel, pCurrAcc, pNextAcc, numNodesX, dx, dt);

    // Copy host -> device
    alpaka::memcpy(queue, uCurrBufAcc, uCurrBufHost);
    // Copy to the buffer for next as well to have boundary values set
    alpaka::memcpy(queue, uNextBufAcc, uCurrBufAcc);
    alpaka::wait(queue);

    for(uint32_t step = 0; step < numTimeSteps; step++)
    {
        // Compute next values
        alpaka::exec<Acc>(queue, workDiv, heatEqKernel, pCurrAcc, pNextAcc, numNodesX, dx, dt);

        // We assume the boundary conditions are constant and so these values
        // do not need to be updated.
        // So we just swap next to curr (shallow copy)
        std::swap(pCurrAcc, pNextAcc);
    }

    // Copy device -> host
    alpaka::memcpy(queue, uNextBufHost, uNextBufAcc);
    alpaka::wait(queue);

    // Calculate error
    double maxError = 0.0;
    for(uint32_t i = 0; i < numNodesX; i++)
    {
        auto const error = std::abs(pNextHost[i] - exactSolution(i * dx, tMax));
        maxError = std::max(maxError, error);
    }

    double const errorThreshold = 1e-5;
    bool resultCorrect = (maxError < errorThreshold);
    if(resultCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: error = " << maxError << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
    }
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
