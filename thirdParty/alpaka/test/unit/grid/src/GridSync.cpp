/* Copyright 2026 Mykhailo Varvarin, Maria Michailidi, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/grid/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

class GridSyncTestKernel
{
public:
    static constexpr std::uint8_t blockThreadExtentPerDim()
    {
        return 2u;
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, T* array) const -> void
    {
        using Idx = alpaka::Idx<TAcc>;

        // Get the index of the current thread within the grid and the grid extent and map them to 1D.
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdx1D = alpaka::mapIdx<1u>(gridThreadIdx, gridThreadExtent)[0u];
        auto const gridThreadExtent1D = gridThreadExtent.prod();


        // Synchronize the grid multiple times, to check that the grid barrier can be reused.
        for(auto round = static_cast<Idx>(0u); round < static_cast<Idx>(3u); ++round)
        {
            // Write the thread index into the global array.
            array[gridThreadIdx1D] = static_cast<T>(gridThreadIdx1D + round * gridThreadExtent1D);

            // Synchronize the threads in the grid.
            alpaka::syncGridThreads(acc);

            // All other threads within the grid should now have written their index into the global memory.
            for(auto i = static_cast<Idx>(0u); i < gridThreadExtent1D; ++i)
            {
                ALPAKA_CHECK(*success, static_cast<Idx>(array[i]) == i + round * gridThreadExtent1D);
            }

            // Synchronize again, before any thread overwrites the array in the next round.
            alpaka::syncGridThreads(acc);
        }
    }
};

//! The launch configuration of a cooperative kernel: the number of threads per block, and the maximum number of blocks
//! that can be active at the same time with that block size.
template<typename TDim, typename TIdx>
struct CooperativeLaunchConfig
{
    alpaka::Vec<TDim, TIdx> threadsPerBlock;
    int maxBlocks;
};

//! Query the launch configuration of the cooperative GridSyncTestKernel. On the GPU back-ends, use the native CUDA,
//! HIP or SYCL functions to get the maximum block size for the kernel, along the innermost dimension, and the maximum
//! number of concurrent blocks with that block size. On the other back-ends, use a small block, and the largest number
//! of blocks that can be active at the same time, as given by alpaka::getMaxActiveBlocks().
template<typename TAcc>
auto nativeCooperativeLaunchConfig(alpaka::Dev<TAcc> const& dev)
    -> CooperativeLaunchConfig<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>>
{
    using Tag [[maybe_unused]] = alpaka::AccToTag<TAcc>;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;
    using Config = CooperativeLaunchConfig<Dim, Idx>;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    if constexpr(std::is_same_v<Tag, alpaka::TagGpuCudaRt>)
    {
        auto const kernel = alpaka::detail::kernelName<GridSyncTestKernel, TAcc, bool*, Idx*>;
        int minGridSize = 0;
        int blockSize = 0;
        REQUIRE(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0) == cudaSuccess);
        int blocksPerMultiprocessor = 0;
        REQUIRE(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor, kernel, blockSize, 0)
            == cudaSuccess);
#    ifdef __CUDACC_DEBUG__
        // as in alpaka::getMaxActiveBlocks(), use at most one block per multiprocessor in device-debug mode
        blocksPerMultiprocessor = std::min(blocksPerMultiprocessor, 1);
#    endif
        int multiprocessors = 0;
        REQUIRE(
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, alpaka::getNativeHandle(dev))
            == cudaSuccess);
        // use the maximum block size along the innermost dimension
        auto threadsPerBlock = alpaka::Vec<Dim, Idx>::all(1);
        threadsPerBlock[Dim::value - 1] = static_cast<Idx>(blockSize);
        return Config{threadsPerBlock, blocksPerMultiprocessor * multiprocessors};
    }
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    if constexpr(std::is_same_v<Tag, alpaka::TagGpuHipRt>)
    {
        auto const kernel = alpaka::detail::kernelName<GridSyncTestKernel, TAcc, bool*, Idx*>;
        int minGridSize = 0;
        int blockSize = 0;
        REQUIRE(hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0) == hipSuccess);
        int blocksPerMultiprocessor = 0;
        REQUIRE(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor, kernel, blockSize, 0)
            == hipSuccess);
        int multiprocessors = 0;
        REQUIRE(
            hipDeviceGetAttribute(
                &multiprocessors,
                hipDeviceAttributeMultiprocessorCount,
                alpaka::getNativeHandle(dev))
            == hipSuccess);
        // use the maximum block size along the innermost dimension
        auto threadsPerBlock = alpaka::Vec<Dim, Idx>::all(1);
        threadsPerBlock[Dim::value - 1] = static_cast<Idx>(blockSize);
        return Config{threadsPerBlock, blocksPerMultiprocessor * multiprocessors};
    }
#endif

#if defined(ALPAKA_ACC_SYCL_ENABLED)
    if constexpr(alpaka::concepts::Tag<Tag> && std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevGenericSycl<Tag>>)
    {
        auto const [device, context] = dev.getNativeHandle();
        sycl::queue queue{context, device};
        auto const bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(context, {device});
        auto const kernel
            = bundle.template get_kernel<alpaka::detail::SyclKernel<GridSyncTestKernel, Dim, Idx, bool*, Idx*>>();
        // the maximum work-group size for the kernel on this device
        auto const blockSize = kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(device);
        // the kernel uses at least 1 byte of dynamic local memory, and the static shared memory, see
        // TaskKernelGenericSycl::operator()
        [[maybe_unused]] auto const localMemBytes = std::size_t{1} + alpaka::detail::syclStaticSharedMemBytes;
        auto blockExtent = alpaka::Vec<Dim, Idx>::all(1);
        blockExtent[Dim::value - 1] = static_cast<Idx>(blockSize);
        [[maybe_unused]] auto const workGroupSize = alpaka::detail::syclWorkGroupSize(blockExtent);
#    if ALPAKA_COMP_ICPX >= ALPAKA_VERSION_NUMBER(2025, 1, 0)
        auto const maxBlocks = kernel.template ext_oneapi_get_info<
            sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_groups>(
            queue,
            workGroupSize,
            localMemBytes);
#    else
        auto const maxBlocks = kernel.template ext_oneapi_get_info<
            sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_group_sync>(queue);
#    endif
        return Config{blockExtent, static_cast<int>(maxBlocks)};
    }
#endif

    // On the other back-ends, use a small block, and the largest number of blocks that can be active at the same time.
    auto const threadsPerBlock = alpaka::elementwise_min(
        alpaka::getAccDevProps<TAcc>(dev).m_blockThreadExtentMax,
        alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(GridSyncTestKernel::blockThreadExtentPerDim())));
    int const maxBlocks = alpaka::getMaxActiveBlocks<TAcc>(
        dev,
        GridSyncTestKernel{},
        threadsPerBlock,
        alpaka::Vec<Dim, Idx>::all(1),
        static_cast<bool*>(nullptr),
        static_cast<Idx*>(nullptr));
    return Config{threadsPerBlock, maxBlocks};
}

TEMPLATE_LIST_TEST_CASE("synchronize", "[gridSync]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    INFO(alpaka::getAccName<Acc>());
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    // Select the first device available on a system, for the chosen accelerator
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = getDevByIdx(platformAcc, 0u);

    // Check if accelerator supports cooperative launch, don't do anything if it doesn't
    if(alpaka::trait::GetAccDevProps<Acc>::getAccDevProps(devAcc).m_cooperativeLaunch)
    {
        // Use the largest block size, and the largest number of blocks that can be active at the same time.
        auto const config = nativeCooperativeLaunchConfig<Acc>(devAcc);
        auto const threadsPerBlock = config.threadsPerBlock;
        int const maxBlocks = config.maxBlocks;
        INFO("threads per block: " << threadsPerBlock << ", maximum blocks: " << maxBlocks);
        REQUIRE(threadsPerBlock.prod() > 0);
        REQUIRE(maxBlocks > 0);

        auto const elementsPerThread = alpaka::Vec<Dim, Idx>::all(1);

        GridSyncTestKernel kernel;

        // Test both the largest grid, and a grid with fewer blocks than can be active at the same time.
        for(int const numBlocks : {maxBlocks, (maxBlocks + 1) / 2})
        {
            INFO("blocks per grid: " << numBlocks << " (maximum " << maxBlocks << ")");
            auto blocksPerGrid = alpaka::Vec<Dim, Idx>::all(1);
            blocksPerGrid[0] = static_cast<Idx>(numBlocks);

            // Allocate memory on the device, after the size of the grid is known.
            alpaka::Vec<alpaka::DimInt<1>, Idx> bufferExtent{
                blocksPerGrid.prod() * threadsPerBlock.prod() * elementsPerThread.prod()};
            auto deviceMemory = alpaka::allocBuf<Idx, Idx>(devAcc, bufferExtent);

            constexpr bool IsCooperative = true;
            alpaka::test::KernelExecutionFixture<Acc, IsCooperative> fixture(
                alpaka::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread});

            CHECK(fixture(kernel, alpaka::getPtrNative(deviceMemory)));
        }
    }
}
