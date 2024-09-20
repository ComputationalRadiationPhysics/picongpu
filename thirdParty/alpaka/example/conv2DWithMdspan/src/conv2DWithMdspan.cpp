/* Copyright 2023 Mehmet Yusufoglu, Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
// Needed for running example for all backends available; one by one
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

//! Convolution Example using Mdspan structure to pass multi-dimensional data to the kernel
//!
//! A 2D Convolutional filter applied to a matrix. Mdspan data structure is used, therefore pitch and size values are
//! not needed to be passed to the kernel. Results can be tested by comparing with the results of the Matlab call: Y =
//! filter2(FilterMatrix,InputMatrix,'same');

/**
 * @brief 2D Convolutional Filter using only global memory for the input-matrix and the filter-matrix
 */
struct ConvolutionKernelMdspan2D
{
    //! \tparam TAcc Accelerator type
    //! \tparam TElem The input-matrix and filter-matrix element type
    //! \param acc Accelerator
    //! \param input Input matrix
    //! \param output Output matrix
    //! \param filter Filter-matrix

    template<typename TAcc, typename MdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, MdSpan const input, MdSpan output, MdSpan const filter) const
        -> void
    {
        static_assert(
            alpaka::Dim<TAcc>::value == 2u,
            "The accelerator used for the Alpaka Kernel has to be 2 dimensional!");

        auto matrixWidth = input.extent(0);
        auto matrixHeight = input.extent(1);
        // Filter matrix is square
        int32_t filterWidth = filter.extent(0);

        // Get thread index, the center of filter-matrix is positioned to the item on this index.
        int32_t const row = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        int32_t const col = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        // The convolutional filter-matrix applied to the input matrix, it's position is row and col. If some of the
        // items of the filter are outside the matrix, those are not taken into calculation or they are assumed zero.
        if(col < matrixWidth && row < matrixHeight)
        {
            float pValue{0.0f};
            for(int32_t fRow = 0; fRow < filterWidth; fRow++)
            {
                for(int32_t fCol = 0; fCol < filterWidth; fCol++)
                {
                    // Position of input matrix element to be multiplied with the corresponding element at filter
                    auto const exactRow = row - filterWidth / 2 + fRow;
                    auto const exactCol = col - filterWidth / 2 + fCol;
                    if(exactRow >= 0 && exactRow < matrixHeight && exactCol >= 0 && exactCol < matrixWidth)
                    {
                        pValue += filter(fRow, fCol) * input(exactRow, exactCol);
                    }
                }
            }
            output(row, col) = pValue;
        }
    }
};

auto FuzzyEqual(float a, float b) -> bool
{
    return std::fabs(a - b) < std::numeric_limits<float>::epsilon() * 1000.0f;
}

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the index domain
    using Dim = alpaka::DimInt<2>;
    // Index type
    using Idx = std::uint32_t;
    using Vec = alpaka::Vec<Dim, Idx>;
    // Define the accelerator
    using DevAcc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using QueueAcc = alpaka::Queue<DevAcc, alpaka::NonBlocking>;

    using DataType = float;
    static constexpr Idx filterWidth = 5;
    static constexpr Idx matrixWidth = 128;
    static constexpr Idx matrixHeight = 128;

    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<DevAcc>() << std::endl;

    auto const devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    // Select a device from the accelerator
    auto const platformAcc = alpaka::Platform<DevAcc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    // Create a queue on the device
    QueueAcc queueAcc(devAcc);
    // Define the 2D extent (dimensions)
    Vec const extent(static_cast<Idx>(matrixWidth), static_cast<Idx>(matrixHeight));

    //
    // Input vector allocation and copy to device buffer
    //
    std::vector<DataType> bufInputHost1D(extent.prod(), 1);
    // Use increasing values as input
    std::iota(bufInputHost1D.begin(), bufInputHost1D.end(), 1.0f);
    for(DataType& element : bufInputHost1D)
    {
        element /= matrixWidth;
    }
    // Create 2D view
    auto bufInputHostView = alpaka::createView(devHost, bufInputHost1D.data(), extent);

    // Input buffer at device
    auto bufInputAcc = alpaka::allocBuf<DataType, Idx>(devAcc, extent);
    // Copy input view from host to device by copying to alpaka buffer type
    alpaka::memcpy(queueAcc, bufInputAcc, bufInputHostView);
    alpaka::wait(queueAcc);
    //
    //  Output buffer allocation at device
    //
    auto outputDeviceMemory = alpaka::allocBuf<DataType, Idx>(devAcc, extent);

    //  Prepare convolution filter at host
    //
    std::vector<DataType> const filter = {0.11, 0.12, 0.13, 0.14, 0.15, 0.21, 0.22, 0.23, 0.24, 0.25, 0.31, 0.32, 0.33,
                                          0.34, 0.35, 0.41, 0.42, 0.43, 0.44, 0.45, 0.51, 0.52, 0.53, 0.54, 0.55};

    Vec const filterExtent(static_cast<Idx>(filterWidth), static_cast<Idx>(filterWidth));
    // Create 2D view from std::vector in order to use in alpaka::memcpy
    auto bufFilterHostView = alpaka::createView(devHost, filter.data(), filterExtent);

    // The buffer for the filter data at device
    auto bufFilterAcc = alpaka::allocBuf<DataType, Idx>(devAcc, filterExtent);
    // Copy input view from host to device by copying to alpaka buffer type
    alpaka::memcpy(queueAcc, bufFilterAcc, bufFilterHostView);
    alpaka::wait(queueAcc);

    //  Construct kernel object
    ConvolutionKernelMdspan2D convolutionKernel2D;

    //   Let alpaka calculate good block and grid sizes given our full problem extent.
    alpaka::KernelCfg<DevAcc> const kernelCfg = {extent, Vec::ones()};
    auto const workDiv = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        convolutionKernel2D,
        alpaka::experimental::getMdSpan(bufInputAcc),
        alpaka::experimental::getMdSpan(outputDeviceMemory),
        alpaka::experimental::getMdSpan(bufFilterAcc));


    // Run the kernel, pass 3 arrays as 2D mdspans
    alpaka::exec<DevAcc>(
        queueAcc,
        workDiv,
        convolutionKernel2D,
        alpaka::experimental::getMdSpan(bufInputAcc),
        alpaka::experimental::getMdSpan(outputDeviceMemory),
        alpaka::experimental::getMdSpan(bufFilterAcc));

    // Allocate memory on host to receive the resulting matrix as an array
    auto resultGpuHost = alpaka::allocBuf<DataType, Idx>(devHost, extent);
    // Copy result from device memory to host
    alpaka::memcpy(queueAcc, resultGpuHost, outputDeviceMemory, extent);
    alpaka::wait(queueAcc);

    //  Print results
    std::cout << "Convolution filter kernel ConvolutionKernelMdspan2D.\n";
    std::cout << "Matrix Size:" << matrixWidth << "x" << matrixHeight << ", Filter Size:" << filterWidth << "x"
              << filterWidth << "\n";

    // Print 2D output as 1D
    //  for(size_t i{0}; i < matrixWidth * matrixHeight; ++i)
    //  {
    //     std::cout << "output[" << i << "]:" << std::setprecision(6) << *(std::data(resultGpuHost) + i) << std::endl;
    //  }

    // Print output using MdSpan
    for(size_t i{0}; i < matrixHeight; ++i)
    {
        for(size_t j{0}; j < matrixWidth; ++j)
        {
            std::cout << "outputMdSpan[" << i << "," << j << "]:" << std::setprecision(6)
                      << alpaka::experimental::getMdSpan(resultGpuHost)(i, j) << std::endl;
        }
    }

    // Expected array of sampled results
    std::vector<DataType> const expectedOutput{
        4.622344e+00,
        1.106426e+02,
        2.162168e+02,
        3.217910e+02,
        4.273652e+02,
        4.199258e+02,
        6.385137e+02,
        7.440879e+02,
        8.496621e+02,
        9.552363e+02,
        4.390715e+02};
    // Select samples from output to check results
    size_t const numberOfSamples{10};
    size_t const samplePeriod{matrixWidth * matrixHeight / numberOfSamples};
    bool allEqual{true};
    for(size_t i{0}; i < numberOfSamples; ++i)
    {
        // Compare with the reference results, select one from every samplePeriod element
        bool fuzzyEqual = FuzzyEqual(*(std::data(resultGpuHost) + i * samplePeriod), expectedOutput[i]);
        if(!fuzzyEqual)
            std::cout << *(std::data(resultGpuHost) + i * samplePeriod) << " " << expectedOutput[i] << std::endl;
        allEqual = allEqual && fuzzyEqual;
    }
    if(!allEqual)
    {
        std::cout << "Error: Some 2D convolution results doesn't match!\n";
        return EXIT_FAILURE;
    }
    std::cout << "Sampled result checks are correct!\n";
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
