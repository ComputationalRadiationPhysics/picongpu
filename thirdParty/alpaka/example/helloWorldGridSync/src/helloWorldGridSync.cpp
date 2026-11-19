/* Copyright 2026 Mykhailo Varvarin, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <type_traits>

//! Hello world kernel, utilizing grid synchronization.
//! Prints hello world from a thread, performs grid sync.
//! and prints the sum of indixes of this thread and the opposite thread (the sums have to be the same).
//! Prints an error if sum is incorrect.
struct HelloWorldKernel
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, size_t* array, bool* success) const
    {
        // Get index of the current thread in the grid and the total number of threads.
        size_t gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        if(alpaka::oncePerGrid(acc))
        {
            printf("Hello, World from alpaka thread %lu!\n", gridThreadIdx);
        }

        // Write the index of the thread to array.
        array[gridThreadIdx] = gridThreadIdx;

        // Perform grid synchronization.
        alpaka::syncGridThreads(acc);

        // Get the index of the thread from the opposite side of 1D array.
        size_t gridThreadIdxOpposite = array[gridThreadExtent - gridThreadIdx - 1];

        // Sum them.
        size_t sum = gridThreadIdx + gridThreadIdxOpposite;

        // Get the expected sum.
        size_t expectedSum = gridThreadExtent - 1;

        // Print the result and signify an error if the grid synchronization fails.
        if(sum != expectedSum)
        {
            *success = false;
            printf(
                "After grid sync, this thread is %lu, thread on the opposite side is %lu. Their sum is %lu, expected: "
                "%lu.%s",
                gridThreadIdx,
                gridThreadIdxOpposite,
                sum,
                expectedSum,
                sum == expectedSum ? "\n" : " ERROR: the sum is incorrect.\n");
        }
    }
};

//! The launch configuration of a cooperative kernel: the number of threads per block, and the maximum number of blocks
//! that can be active at the same time with that block size.
struct CooperativeLaunchConfig
{
    std::size_t threadsPerBlock;
    std::size_t maxBlocks;
};

//! Query the launch configuration of the cooperative HelloWorldKernel. On the GPU back-ends, use the native CUDA, HIP
//! or SYCL functions to get the maximum block size for the kernel, and the maximum number of concurrent blocks with
//! that block size. On the other back-ends, use the block size suggested by alpaka::getValidWorkDiv(), and the largest
//! number of blocks that can be active at the same time, as given by alpaka::getMaxActiveBlocks().
//! If a native query fails, the maximum number of blocks is 0.
template<typename TAcc>
auto nativeCooperativeLaunchConfig(
    alpaka::Dev<TAcc> const& dev,
    alpaka::Idx<TAcc> totalSize,
    alpaka::Idx<TAcc> elementsPerThread) -> CooperativeLaunchConfig
{
    using Tag [[maybe_unused]] = alpaka::AccToTag<TAcc>;
    using Dim [[maybe_unused]] = alpaka::Dim<TAcc>;
    using Idx [[maybe_unused]] = alpaka::Idx<TAcc>;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<Tag, alpaka::TagGpuCudaRt>)
    {
        auto const kernel = alpaka::detail::kernelName<HelloWorldKernel, TAcc, size_t*, bool*>;
        int minGridSize = 0;
        int blockSize = 0;
        int blocksPerMultiprocessor = 0;
        int multiprocessors = 0;
        if(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0) != cudaSuccess
           || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor, kernel, blockSize, 0)
                  != cudaSuccess
           || cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, alpaka::getNativeHandle(dev))
                  != cudaSuccess)
            return CooperativeLaunchConfig{1, 0};
#    ifdef __CUDACC_DEBUG__
        // as in alpaka::getMaxActiveBlocks(), use at most one block per multiprocessor in device-debug mode
        blocksPerMultiprocessor = std::min(blocksPerMultiprocessor, 1);
#    endif
        return CooperativeLaunchConfig{
            static_cast<std::size_t>(blockSize),
            static_cast<std::size_t>(blocksPerMultiprocessor) * static_cast<std::size_t>(multiprocessors)};
    }
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    if constexpr(std::is_same_v<Tag, alpaka::TagGpuHipRt>)
    {
        auto const kernel = alpaka::detail::kernelName<HelloWorldKernel, TAcc, size_t*, bool*>;
        int minGridSize = 0;
        int blockSize = 0;
        int blocksPerMultiprocessor = 0;
        int multiprocessors = 0;
        if(hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0) != hipSuccess
           || hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor, kernel, blockSize, 0)
                  != hipSuccess
           || hipDeviceGetAttribute(
                  &multiprocessors,
                  hipDeviceAttributeMultiprocessorCount,
                  alpaka::getNativeHandle(dev))
                  != hipSuccess)
            return CooperativeLaunchConfig{1, 0};
        return CooperativeLaunchConfig{
            static_cast<std::size_t>(blockSize),
            static_cast<std::size_t>(blocksPerMultiprocessor) * static_cast<std::size_t>(multiprocessors)};
    }
#endif

#if defined(ALPAKA_ACC_SYCL_ENABLED)
    if constexpr(alpaka::concepts::Tag<Tag> && std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevGenericSycl<Tag>>)
    {
        auto const [device, context] = dev.getNativeHandle();
        sycl::queue queue{context, device};
        auto const bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(context, {device});
        auto const kernel
            = bundle.template get_kernel<alpaka::detail::SyclKernel<HelloWorldKernel, Dim, Idx, size_t*, bool*>>();
        // the maximum work-group size for the kernel on this device
        auto const blockSize = kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(device);
        // the kernel uses at least 1 byte of dynamic local memory, and the static shared memory
        [[maybe_unused]] auto const localMemBytes = std::size_t{1} + alpaka::detail::syclStaticSharedMemBytes;
#    if ALPAKA_COMP_ICPX >= ALPAKA_VERSION_NUMBER(2025, 1, 0)
        auto const maxBlocks = kernel.template ext_oneapi_get_info<
            sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_groups>(
            queue,
            sycl::range<1>{blockSize},
            localMemBytes);
#    else
        auto const maxBlocks = kernel.template ext_oneapi_get_info<
            sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_group_sync>(queue);
#    endif
        return CooperativeLaunchConfig{blockSize, maxBlocks};
    }
#endif

    // On the other back-ends, use the block size suggested by alpaka, and the largest number of blocks that can be
    // active at the same time.
    alpaka::KernelCfg<TAcc> config
        = {totalSize, elementsPerThread, false, alpaka::GridBlockExtentSubDivRestrictions::Unrestricted};
    auto const workDiv = alpaka::getValidWorkDiv(config, dev, HelloWorldKernel{}, (size_t*) nullptr, (bool*) nullptr);
    int const maxBlocks = alpaka::getMaxActiveBlocks<TAcc>(
        dev,
        HelloWorldKernel{},
        workDiv.m_blockThreadExtent,
        workDiv.m_threadElemExtent,
        (size_t*) nullptr,
        (bool*) nullptr);
    return CooperativeLaunchConfig{
        static_cast<std::size_t>(workDiv.m_blockThreadExtent[0]),
        static_cast<std::size_t>(std::max(0, maxBlocks))};
}

// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // Define the accelerator
    // For simplicity this examples always uses 1 dimensional indexing, and index type size_t
    using Acc = alpaka::TagToAcc<TAccTag, alpaka::DimInt<1>, std::size_t>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    // Define dimensionality and type of indices to be used in kernels
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Scalar = alpaka::Vec<alpaka::DimInt<0u>, Idx>;

    // Select the first device available on a system, for the chosen accelerator
    auto const platform = alpaka::Platform<Acc>{};
    auto const device = getDevByIdx(platform, 0u);

    // Select CPU host
    constexpr auto platformHost = alpaka::Platform<alpaka::DevCpu>{};
    auto const host = getDevByIdx(platformHost, 0u);

    // Define type for a queue with requested properties: Blocking.
    using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
    // Create a queue for the device.
    auto queue = Queue{device};

    // Grid synchronization requires a device that supports cooperative kernels.
    if(!alpaka::getAccDevProps<Acc>(device).m_cooperativeLaunch)
    {
        std::cout << "The device does not support cooperative kernels, skipping." << std::endl;
        return EXIT_SUCCESS;
    }

    // Instantiate the kernel object.
    HelloWorldKernel helloWorldKernel{};

    const Idx totalSize = 1024 * 1024;
    const Idx elementsPerThread = 1;
    // Query the largest block size, and the largest number of blocks that can be active at the same time.
    auto const launchConfig = nativeCooperativeLaunchConfig<Acc>(device, totalSize, elementsPerThread);
    Idx const activeBlocks = static_cast<Idx>(launchConfig.maxBlocks);
    alpaka::WorkDivMembers<Dim, Idx> grid{Idx{1}, static_cast<Idx>(launchConfig.threadsPerBlock), elementsPerThread};
    grid.m_gridBlockExtent[0] = (totalSize + grid.m_blockThreadExtent[0] - 1) / grid.m_blockThreadExtent[0];
    std::cout << "Threads per block: " << grid.m_blockThreadExtent[0] << std::endl;
    std::cout << "Maximum blocks for the kernel: " << activeBlocks << std::endl;
    if(activeBlocks == 0)
    {
        std::cerr << "The kernel cannot be launched as a cooperative kernel on this device." << std::endl;
        return EXIT_FAILURE;
    }

    // Limit the number of blocks to the value supported by the device.
    if(grid.m_gridBlockExtent[0] > activeBlocks)
    {
        grid.m_gridBlockExtent[0] = activeBlocks;
    }

    // Allocate memory on the device.
    auto d_array = alpaka::allocBuf<size_t, Idx>(device, Idx{grid.m_gridBlockExtent[0] * grid.m_blockThreadExtent[0]});

    // Allocate the result value.
    auto d_result = alpaka::allocBuf<bool, Idx>(device, Scalar{});
    alpaka::memset(queue, d_result, static_cast<std::uint8_t>(true));

    // Create a task to run the kernel.
    // Note the cooperative kernel specification: only cooperative kernels can
    // perform grid synchronization.
    auto taskRunKernel
        = alpaka::createTaskCooperativeKernel<Acc>(grid, helloWorldKernel, d_array.data(), d_result.data());

    // Enqueue the kernel execution task.
    alpaka::enqueue(queue, taskRunKernel);

    // Copy the result value to the host
    auto h_result = alpaka::allocBuf<bool, Idx>(host, Scalar{});
    alpaka::memcpy(queue, h_result, d_result);
    alpaka::wait(queue);

    return *h_result ? EXIT_SUCCESS : EXIT_FAILURE;
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
