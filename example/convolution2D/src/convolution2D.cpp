/* Copyright 2023 Mehmet Yusufoglu, Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

//! Convolution Example
//!
//! A 2D Convolutional filter applied to a matrix. The first kernel, ConvolutionKernel2DGlobalMemory, uses only global
//! memory. The second kernel ConvolutionKernel2DSharedMemory uses tiling and shared memory. Block size is assumed to
//! be equal to the tile size. First, the tile is copied to shared memory, since an element in a tile is accessed many
//! times; using the shared memory for the main matrix data increases performance. Each block works on the domain of
//! one tile. But at the border of the tile, some external matrix values are needed (at the border with another tile)
//! then those matrix values are taken from the global memory.
//! Results can be tested by comparing with the results of the Matlab call: Y =
//! filter2(FilterMatrix,InputMatrix,'same');

/**
 * @brief 2D Convolutional Filter using only global memory for the input-matrix and the filter-matrix
 */
struct ConvolutionKernel2DGlobalMemory
{
    //! \tparam TAcc Accelerator type
    //! \tparam TElem The input-matrix and filter-matrix element type
    //! \param acc Accelerator
    //! \param input Input matrix
    //! \param output Output matrix
    //! \param matrixWidth Input matrix width
    //! \param matrixHeight Input matrix height
    //! \param filter Filter-matrix
    //! \param filterWidth Filter-matrix width
    //! \param intputWidthAllocated Input-matrix width allocated (possibly larger than normal width due to paddding)
    //! \param filterWidthAllocated Filter-matrix width allocated (possibly larger than normal width due to paddding

    template<typename TAcc, typename TElem>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const input,
        TElem* output,
        int32_t const matrixWidth,
        int32_t const matrixHeight,
        TElem const* const filter,
        int32_t const filterWidth,
        int32_t const intputWidthAllocated,
        int32_t const filterWidthAllocated) const -> void
    {
        // Get thread index, the center of filter-matrix is positioned to the item on this index.
        auto const [row, col] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        // Block index with respect to thread
        auto const [blockThreadY, blockThreadX] = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);

        // The convolutional filter-matrix applied to the input matrix, it's position is row and col. If some of the
        // items of the filter are outside the matrix, those are not taken into calculation or they are assumed zero.
        if(col < matrixWidth && row < matrixHeight)
        {
            TElem pValue{0.0f};
            for(int32_t fRow = 0; fRow < filterWidth; fRow++)
            {
                for(int32_t fCol = 0; fCol < filterWidth; fCol++)
                {
                    // Position of input matrix element to be multiplied with the corresponding element at filter
                    auto const exactRow = static_cast<int32_t>(row) - filterWidth / 2 + fRow;
                    auto const exactCol = static_cast<int32_t>(col) - filterWidth / 2 + fCol;
                    if(exactRow >= 0 && exactRow < matrixHeight && exactCol >= 0 && exactCol < matrixWidth)
                    {
                        pValue += filter[fRow * filterWidthAllocated + fCol]
                                  * input[exactRow * intputWidthAllocated + exactCol];
                    }
                }
                output[row * matrixWidth + col] = pValue;
            }
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
template<alpaka::concepts::Tag TAccTag>
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

    static_assert(
        alpaka::Dim<DevAcc>::value == 2u,
        "The accelerator used for the Alpaka Kernel has to be 2 dimensional!");

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

    // Calculate the allocated width, due to padding it might be larger then the matrix width
    auto const intputWidthAllocated = [&]() -> Idx const
    {
        // Calculate pitch: The size of one line in bytes including padding.
        auto const rowPitchInput{alpaka::getPitchesInBytes(bufInputAcc)[0]};
        return static_cast<Idx>(rowPitchInput / sizeof(DataType));
    }();

    //
    //  Output buffer allocation at device
    //
    alpaka::Vec<alpaka::DimInt<1u>, Idx> const extent1D(matrixHeight * matrixWidth);
    auto outputDeviceMemory = alpaka::allocBuf<DataType, Idx>(devAcc, extent1D);

    //   Prepare convolution filter
    //
    std::vector<DataType> const filter = {0.11, 0.12, 0.13, 0.14, 0.15, 0.21, 0.22, 0.23, 0.24, 0.25, 0.31, 0.32, 0.33,
                                          0.34, 0.35, 0.41, 0.42, 0.43, 0.44, 0.45, 0.51, 0.52, 0.53, 0.54, 0.55};

    Vec const filterExtent(static_cast<Idx>(filterWidth), static_cast<Idx>(filterWidth));
    // Create 2D view
    auto bufFilterHostView = alpaka::createView(devHost, filter.data(), filterExtent);

    // Filter buffer at device
    auto bufFilterAcc = alpaka::allocBuf<DataType, Idx>(devAcc, filterExtent);
    // Copy input view from host to device by copying to alpaka buffer type
    alpaka::memcpy(queueAcc, bufFilterAcc, bufFilterHostView);
    alpaka::wait(queueAcc);

    // Calculate the allocated width, due to padding it might be larger then the matrix width
    auto const filterWidthAllocated = [&]() -> Idx const
    {
        // Calculate pitch: The size of one line in bytes including padding.
        auto const rowPitchFilter{alpaka::getPitchesInBytes(bufFilterAcc)[0]};
        return static_cast<Idx>(rowPitchFilter / sizeof(DataType));
    }();

    ConvolutionKernel2DGlobalMemory convolutionKernel2D;

    alpaka::KernelCfg<DevAcc> kernelCfg = {extent, Vec::ones()};

    //   Let alpaka calculate good block and grid sizes given our full problem extent.
    auto const workDiv = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        convolutionKernel2D,
        std::data(bufInputAcc),
        std::data(outputDeviceMemory),
        matrixWidth,
        matrixHeight,
        std::data(bufFilterAcc),
        filterWidth,
        intputWidthAllocated,
        filterWidthAllocated);

    // Run the kernel
    alpaka::exec<DevAcc>(
        queueAcc,
        workDiv,
        convolutionKernel2D,
        std::data(bufInputAcc),
        std::data(outputDeviceMemory),
        matrixWidth,
        matrixHeight,
        std::data(bufFilterAcc),
        filterWidth,
        intputWidthAllocated,
        filterWidthAllocated);

    // Allocate memory on host to receive the resulting matrix as an array
    auto resultGpuHost = alpaka::allocBuf<DataType, Idx>(devHost, extent1D);
    // Copy result from device memory to host
    alpaka::memcpy(queueAcc, resultGpuHost, outputDeviceMemory, extent1D);
    alpaka::wait(queueAcc);

    //  Print results
    //
    std::string const kernelType{
        std::is_same<decltype(convolutionKernel2D), ConvolutionKernel2DGlobalMemory>::value
            ? "ConvolutionKernel2DGlobalMemory"
            : "ConvolutionKernel2DSharedMemory"};

    std::cout << "Convolution filter kernel:" << kernelType << "\n";
    std::cout << "Matrix Size:" << matrixWidth << "x" << matrixHeight << ", Filter Size:" << filterWidth << "x"
              << filterWidth << "\n";

    // Print output
    for(size_t i{0}; i < matrixWidth * matrixHeight; ++i)
    {
        std::cout << "output[" << i << "]:" << std::setprecision(6) << resultGpuHost[i] << std::endl;
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
        bool fuzzyEqual = FuzzyEqual(resultGpuHost[i * samplePeriod], expectedOutput[i]);
        if(!fuzzyEqual)
            std::cout << resultGpuHost[i * samplePeriod] << " " << expectedOutput[i] << std::endl;
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
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
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
