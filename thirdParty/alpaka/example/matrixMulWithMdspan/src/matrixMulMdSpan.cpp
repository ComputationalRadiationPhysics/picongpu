/* Copyright 2024 Mehmet Yusufoglu, Simeon Ehrig, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
// Needed for running example for all backends available; one by one
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <experimental/mdspan>
#include <iostream>

//! Matrix multiplication example by using mdspan data structure

//! Some simple type traits for checking the types
//! isMdspan simply checks if a type is of type std::experimental::mdspan or not
//! Primary template for is_mdspan (defaults to false)
template<typename T>
struct IsMdspan : std::false_type
{
};

//! Specialization for mdspan with four template arguments
template<typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
struct IsMdspan<std::experimental::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>> : std::true_type
{
};

template<typename T>
inline constexpr bool is_mdspan = IsMdspan<T>::value;

// Index type
using Idx = std::size_t;
// Set data type
using DataType = float;

/**
 * @brief Kernel for performing multiplication of two 2D matrices. Each element is computed by a different thread.
 * MdSpan data structure is used to pass the data to and from the kernel.
 */
struct MatrixMulKernel
{
    //! \tparam TAcc Accelerator type
    //! \tparam MdSpan The type of the multidimensional span (mdspan)
    //! \param acc Accelerator
    //! \param A First input matrix
    //! \param B Second input matrix
    //! \param C Output matrix where the result of A * B will be stored
    //! \param K The shared dimension between A and B
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TMdSpan A, TMdSpan B, TMdSpan C) const
    {
        // compile time checks
        static_assert(is_mdspan<TMdSpan>, "The type TMdSpan should be an std mdspan");
        static_assert(TMdSpan::rank() == 2);

        // A is MxK and B is KxN
        auto const K = static_cast<Idx>(A.extent(1));

        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i < C.extent(0) && j < C.extent(1))
        {
            DataType sum = 0.0f;
            for(Idx k = 0; k < K; ++k)
            {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
};

// initialize the matrix
template<typename TMdSpan>
inline void initializeMatrx(TMdSpan& span)
{
    auto const numColumns = span.extent(1);
    for(Idx i = 0; i < span.extent(0); ++i)
    {
        for(Idx j = 0; j < numColumns; ++j)
        {
            // fill with some data
            span(i, j) = static_cast<DataType>(i * numColumns + j);
        }
    }
}

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Set number of dimensions (i.e 2) as a type
    using Dim = alpaka::DimInt<2>;

    // Define matrix dimensions, A is MxK and B is KxN
    Idx const M = 1024;
    Idx const N = 512;
    Idx const K = 1024;

    // Define device and queue
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    Queue queue(devAcc);

    // Define the 2D extents (dimensions)
    Vec const extentA(static_cast<Idx>(M), static_cast<Idx>(K));
    Vec const extentB(static_cast<Idx>(K), static_cast<Idx>(N));
    Vec const extentC(static_cast<Idx>(M), static_cast<Idx>(N));

    // Allocate host memory
    auto bufHostA = alpaka::allocBuf<DataType, Idx>(devHost, extentA);
    auto bufHostB = alpaka::allocBuf<DataType, Idx>(devHost, extentB);
    auto bufHostC = alpaka::allocBuf<DataType, Idx>(devHost, extentC);

    // Create mdspan view for bufHostA and bufHostB using alpaka::experimental::getMdSpan to fill the host buffers
    auto mdHostA = alpaka::experimental::getMdSpan(bufHostA);
    auto mdHostB = alpaka::experimental::getMdSpan(bufHostB);

    // Initialize host matrices
    initializeMatrx(mdHostA);
    initializeMatrx(mdHostB);

    // Allocate device memory
    auto bufDevA = alpaka::allocBuf<DataType, Idx>(devAcc, extentA);
    auto bufDevB = alpaka::allocBuf<DataType, Idx>(devAcc, extentB);
    auto bufDevC = alpaka::allocBuf<DataType, Idx>(devAcc, extentC);

    // Copy data to device, use directly host buffers (not mdspans used to fill the data)
    alpaka::memcpy(queue, bufDevA, bufHostA);
    alpaka::memcpy(queue, bufDevB, bufHostB);
    alpaka::wait(queue);

    // Create mdspan views for device buffers using alpaka::experimental::getMdSpan
    auto mdDevA = alpaka::experimental::getMdSpan(bufDevA);
    auto mdDevB = alpaka::experimental::getMdSpan(bufDevB);
    auto mdDevC = alpaka::experimental::getMdSpan(bufDevC);

    MatrixMulKernel kernel;

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::KernelCfg<Acc> const kernelCfg
        = {extentC, Vec::ones(), false, alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};
    auto const workDiv = alpaka::getValidWorkDiv<Acc>(kernelCfg, devAcc, kernel, mdDevA, mdDevB, mdDevC);

    // Execute the kernel
    alpaka::exec<Acc>(queue, workDiv, kernel, mdDevA, mdDevB, mdDevC);

    // Copy result back to host
    alpaka::memcpy(queue, bufHostC, bufDevC);
    alpaka::wait(queue);

    // Verify the result
    bool success = true;
    auto mdHostC = alpaka::experimental::getMdSpan(bufHostC);
    for(Idx i = 0; i < M; ++i)
    {
        for(Idx j = 0; j < N; ++j)
        {
            DataType expectedValue = 0.0f;
            for(Idx k = 0; k < K; ++k)
            {
                expectedValue += mdHostA(i, k) * mdHostB(k, j);
            }
            if(mdHostC(i, j) != expectedValue)
            {
                success = false;
                break;
            }
        }
    }

    std::cout << "Multiplication of matrices of size " << M << "x" << K << " and " << K << "x" << N << " using mdspan "
              << (success ? "succeeded" : "failed") << "!" << std::endl;
    if(!success)
    {
        return EXIT_FAILURE;
    }

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
