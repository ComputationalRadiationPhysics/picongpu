/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz, Rene Widera

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#pragma once

#include "mallocMC/creationPolicies/FlatterScatter/AccessBlock.hpp"

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/mem/fence/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>

namespace mallocMC::CreationPolicies::FlatterScatterAlloc
{
    /**
     * @class Heap
     * @brief Main interface to our heap memory.
     *
     * This class stores the heap pointer and the heap size and provides the high-level functionality to interact with
     * the memory within kernels. It is wrapped in a thin layer of creation policy to be instantiated as base class of
     * the `DeviceAllocator` for the user.
     *
     * @tparam T_HeapConfig Struct containing information about the heap.
     * @tparam T_HashConfig Struct providing a hash function for scattering and the blockStride property.
     * @tparam T_AlignmentPolicy The alignment policy used in the current configuration.
     */
    template<typename T_HeapConfig, typename T_HashConfig, typename T_AlignmentPolicy>
    struct Heap
    {
        using MyAccessBlock = AccessBlock<T_HeapConfig, T_AlignmentPolicy>;

        static_assert(
            T_HeapConfig::accessblocksize
                < std::numeric_limits<std::make_signed_t<decltype(T_HeapConfig::accessblocksize)>>::max(),
            "Your access block size must be smaller than the maximal value of its signed type because we are using "
            "differences in the code occasionally.");

        static_assert(
            T_HeapConfig::pagesize < std::numeric_limits<std::make_signed_t<decltype(T_HeapConfig::pagesize)>>::max(),
            "Your page size must be smaller than the maximal value of its signed type because we are using "
            "differences in the code occasionally.");

        static_assert(
            T_HeapConfig::accessblocksize == sizeof(MyAccessBlock),
            "The real access block must have the same size as configured in order to make alignment more easily "
            "predictable.");

        size_t heapSize{};
        MyAccessBlock* accessBlocks{};
        uint32_t volatile block = 0U;

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto init() -> void
        {
            for(uint32_t i = 0; i < numBlocks(); ++i)
            {
                accessBlocks[i].init();
            }
        }

        /**
         * @brief Number of access blocks in the heap. This is a runtime quantity because it depends on the given heap
         * size.
         *
         * @return Number of access blocks in the heap.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto numBlocks() const -> uint32_t
        {
            return heapSize / T_HeapConfig::accessblocksize;
        }

        /**
         * @brief The dummy value to indicate the case of no free blocks found.
         *
         * @return An invalid block index for identifying such case.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto noFreeBlockFound() const -> uint32_t
        {
            return numBlocks();
        }

        /**
         * @brief Compute a starting index to search the access blocks for a valid piece of memory.
         *
         * @param blockValue Current starting index to compute the next one from.
         * @param hashValue A hash value to provide some entropy for scattering the requests.
         * @return An index to start search the access blocks from.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto startBlockIndex(
            auto const& /*acc*/,
            uint32_t const blockValue,
            uint32_t const hashValue)
        {
            return ((hashValue % T_HashConfig::blockStride) + (blockValue * T_HashConfig::blockStride)) % numBlocks();
        }

        /**
         * @brief Create a pointer to memory of (at least) `bytes` number of bytes..
         *
         * @param bytes Size of the allocation in number of bytes.
         * @return Pointer to the memory, nullptr if no usable memory was found.
         */
        template<typename AlpakaAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto create(AlpakaAcc const& acc, uint32_t const bytes) -> void*
        {
            auto blockValue = block;
            auto hashValue = T_HashConfig::template hash<T_HeapConfig::pagesize>(acc, bytes);
            auto startIdx = startBlockIndex(acc, blockValue, hashValue);
            return wrappingLoop(
                acc,
                startIdx,
                numBlocks(),
                static_cast<void*>(nullptr),
                [this, bytes, startIdx, &hashValue, blockValue](auto const& localAcc, auto const index) mutable
                {
                    auto ptr = accessBlocks[index].create(localAcc, bytes, hashValue);
                    if(!ptr && index == startIdx)
                    {
                        // This is not thread-safe but we're fine with that. It's just a fuzzy thing to occasionally
                        // increment and it's totally okay if its value is not quite deterministic.
                        if(blockValue == block)
                        {
                            block = blockValue + 1;
                        }
                    }
                    return ptr;
                });
        }

        /**
         * @brief Counterpart free'ing operation to `create`. Destroys the memory at the pointer location.
         *
         * @param pointer A valid pointer created by `create()`.`
         */
        template<typename AlpakaAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto destroy(AlpakaAcc const& acc, void* pointer) -> void
        {
            // indexOf requires the access block size instead of blockSize in case the reinterpreted AccessBlock
            // object is smaller than blockSize.
            auto blockIndex = indexOf(pointer, accessBlocks, sizeof(MyAccessBlock));
            accessBlocks[blockIndex].destroy(acc, pointer);
        }

        /**
         * @brief Queries all access blocks how many chunks of the given chunksize they could allocate. This is
         * single-threaded and NOT THREAD-SAFE but acquiring such distributed information while other threads operate
         * on the heap is of limited value anyways.
         *
         * @param chunkSize Target would-be-created chunk size in number of bytes.
         * @return The number of allocations that would still be possible with this chunk size.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableSlotsDeviceFunction(auto const& acc, uint32_t const chunkSize)
            -> size_t
        {
            // TODO(lenz): Not thread-safe.
            return std::transform_reduce(
                accessBlocks,
                accessBlocks + numBlocks(),
                0U,
                std::plus<size_t>{},
                [&acc, chunkSize](auto& accessBlock) { return accessBlock.getAvailableSlots(acc, chunkSize); });
        }

        /**
         * @brief Forwards to `getAvailableSlotsDeviceFunction` for interface compatibility reasons. See there for
         * details.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableSlotsAccelerator(auto const& acc, uint32_t const chunkSize)
            -> size_t
        {
            return getAvailableSlotsDeviceFunction(acc, chunkSize);
        }

    protected:
        // This class is supposed to be instantiated as a parent for the `DeviceAllocator`.
        Heap() = default;
    };

    constexpr uint32_t defaultBlockSize = 128U * 1024U * 1024U;
    constexpr uint32_t defaultPageSize = 128U * 1024U;

    /**
     * @class DefaultHeapConfig
     * @brief An example configuration for the heap.
     *
     * A heap configuration is supposed to provide the physical dimensions of the objects in the heap (i.e. access
     * block and page) as well as a function that describes how much space you are willing to waste by allowing to
     * allocate larger chunks that necessary.
     *
     * @tparam T_blockSize The size of one access block in bytes.
     * @tparam T_pageSize The size of one page in bytes.
     * @return
     */
    template<
        uint32_t T_blockSize = defaultBlockSize,
        uint32_t T_pageSize = defaultPageSize,
        uint32_t T_wasteFactor = 2U>
    struct DefaultHeapConfig
    {
        static constexpr uint32_t const accessblocksize = T_blockSize;
        static constexpr uint32_t const pagesize = T_pageSize;
        static constexpr uint32_t const wastefactor = T_wasteFactor;
        static constexpr bool const resetfreedpages = true;

        /**
         * @brief Determine whether we want to allow an allocation of numBytes on a page with chunk size `chunkSize`.
         *
         * This function is given the currently requested allocation size numBytes and the set chunk size of a page. It
         * answers the question whether we should consider this page for allocating this memory. It must necessarily
         * return false if chunkSize < numBytes in order to avoid memory corruption. It may return true in cases where
         * chunkSize > numBytes to trade off a bit of wasted memory for a performance boost while searching available
         * memory.
         *
         * @param chunkSize Currently set chunk size of a page in number of bytes.
         * @param numBytes Allocation size in number of bytes.
         * @return true if the algorithm shall consider this page for allocation and false otherwise.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto isInAllowedRange(
            auto const& /*acc*/,
            uint32_t const chunkSize,
            uint32_t const numBytes)
        {
            return (chunkSize >= numBytes && chunkSize <= wastefactor * numBytes);
        }
    };

    /**
     * @class DefaultFlatterScatterHashConfig
     * @brief An example configuration for the hash scattering.
     *
     * A scatter configuration is supposed to provide two pieces of information: A static function called `hash` and
     * the compile-time constant `blockStride`. These are used by the creation policy to scatter the requests for
     * memory within the heap.
     *
     */
    struct DefaultFlatterScatterHashConfig
    {
    public:
        static constexpr uint32_t blockStride = 4;

        /**
         * @brief Hash function to provide entropy for scattering memory requests.
         *
         * @param numBytes Number of bytes requested.
         * @return A hash value.
         */
        // TAcc is to be deduced, so we put it last.
        template<uint32_t T_pageSize, typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto hash(TAcc const& acc, uint32_t const numBytes) -> uint32_t
        {
            uint32_t const relative_offset = warpSize<TAcc> * numBytes / T_pageSize;
            return (
                numBytes * hashingK + hashingDistMP * smid(acc)
                + (hashingDistWP + hashingDistWPRel * relative_offset) * warpid(acc));
        }

    private:
        static constexpr uint32_t hashingK = 38183;
        static constexpr uint32_t hashingDistMP = 17497;
        static constexpr uint32_t hashingDistWP = 1;
        static constexpr uint32_t hashingDistWPRel = 1;
    };

    /**
     * @class InitKernel
     * @brief Kernel to initialise the heap memory.
     *
     * Used by the creation policy during initialisation.
     */
    struct InitKernel
    {
        template<typename T_HeapConfig, typename T_HashConfig, typename T_AlignmentPolicy>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator()(
            auto const& /*unused*/,
            Heap<T_HeapConfig, T_HashConfig, T_AlignmentPolicy>* m_heap,
            void* m_heapmem,
            size_t const m_memsize) const
        {
            m_heap->accessBlocks
                = static_cast<Heap<T_HeapConfig, T_HashConfig, T_AlignmentPolicy>::MyAccessBlock*>(m_heapmem);
            m_heap->heapSize = m_memsize;
            m_heap->init();
        }
    };

} // namespace mallocMC::CreationPolicies::FlatterScatterAlloc

namespace mallocMC::CreationPolicies
{
    /**
     * @class FlatterScatter
     * @brief A creation policy scattering memory requests in a flat hierarchy.
     *
     * This creation policy is a variation on the original ScatterAlloc algorithm and the one previously implemented in
     * mallocMC. It provides a multi-level hierarchy of Heap, AccessBlock and DataPage that is traversed using the
     * metadata held by each level to find a suitable memory location to satisfy the request.
     *
     * It uses a externally provided hash function to compute a single hash value for each request given its requested
     * number of bytes and the accelerator. This is internally used to scatter the requests over the available memory
     * and thereby improve the success rate for multi-threaded requests because different threads will start searching
     * in different locations.
     *
     * Implemented as a thin wrapper around `Heap` that mainly provides interface compatibility with the calling code.
     */
    template<typename T_HeapConfig, typename T_HashConfig, typename T_AlignmentPolicy>
    struct FlatterScatterImpl
    {
        template<typename T>
        using AlignmentAwarePolicy = FlatterScatterAlloc::Heap<T_HeapConfig, T_HashConfig, T>;

        static auto classname() -> std::string
        {
            return "FlatterScatter";
        }

        static constexpr auto const providesAvailableSlots = true;

        /**
         * @brief Check if a pointer returned from `create()` signals out-of-memory.
         *
         * @param pointer Pointer returned by `create()`.
         * @return The boolean answer to this question.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto isOOM(void* pointer, uint32_t const /*unused size*/) -> bool
        {
            return pointer == nullptr;
        }

        /**
         * @brief initialise a raw piece of memory for use by the `Heap`.
         *
         * @param dev The alpaka device.
         * @param queue The alpaka queue.
         * @param heap The pointer to the `Heap` object located on the device.
         * @param pool The pointer to the provided memory pool to be used by the `Heap` object.
         * @param memsize The size of the pool memory in bytes.
         */
        template<typename TAcc>
        static void initHeap(auto& dev, auto& queue, auto* heap, void* pool, size_t memsize)
        {
            using Dim = typename alpaka::trait::DimType<TAcc>::type;
            using Idx = typename alpaka::trait::IdxType<TAcc>::type;
            using VecType = alpaka::Vec<Dim, Idx>;

            auto workDivSingleThread
                = alpaka::WorkDivMembers<Dim, Idx>{VecType::ones(), VecType::ones(), VecType::ones()};
            alpaka::exec<TAcc>(queue, workDivSingleThread, FlatterScatterAlloc::InitKernel{}, heap, pool, memsize);
            alpaka::wait(queue);
        }

        /**
         * @brief Count the number of possible allocation for the given slotSize directly from the host.
         *
         * This method implements the infrastructure to call `getAvailableSlotsDeviceFunction` on the `Heap` class. See
         * there for details, particularly concerning the thread-safety of this.
         *
         * @param dev The alpaka device.
         * @param queue The alpaka queue.
         * @param slotSize The would-be-created memory size in number of bytes.
         * @param heap Pointer to the `Heap` object that's supposed to handle the request.
         * @return The number of allocations that would be successful with this slotSize.
         */
        template<typename AlpakaAcc, typename AlpakaDevice, typename AlpakaQueue, typename T_DeviceAllocator>
        static auto getAvailableSlotsHost(
            AlpakaDevice& dev,
            AlpakaQueue& queue,
            uint32_t const slotSize,
            T_DeviceAllocator* heap) -> unsigned
        {
            using Dim = typename alpaka::trait::DimType<AlpakaAcc>::type;
            using Idx = typename alpaka::trait::IdxType<AlpakaAcc>::type;
            using VecType = alpaka::Vec<Dim, Idx>;

            auto d_slots = alpaka::allocBuf<size_t, uint32_t>(dev, uint32_t{1});
            alpaka::memset(queue, d_slots, 0, uint32_t{1});
            auto d_slotsPtr = alpaka::getPtrNative(d_slots);

            auto getAvailableSlotsKernel = [heap, slotSize, d_slotsPtr] ALPAKA_FN_ACC(AlpakaAcc const& acc) -> void
            { *d_slotsPtr = heap->getAvailableSlotsDeviceFunction(acc, slotSize); };

            alpaka::wait(queue);
            alpaka::exec<AlpakaAcc>(
                queue,
                alpaka::WorkDivMembers<Dim, Idx>{VecType::ones(), VecType::ones(), VecType::ones()},
                getAvailableSlotsKernel);
            alpaka::wait(queue);

            auto const platform = alpaka::Platform<alpaka::DevCpu>{};
            auto const hostDev = alpaka::getDevByIdx(platform, 0);

            auto h_slots = alpaka::allocBuf<size_t, Idx>(hostDev, Idx{1});
            alpaka::memcpy(queue, h_slots, d_slots);
            alpaka::wait(queue);

            return *alpaka::getPtrNative(h_slots);
        }
    };

    template<
        typename T_HeapConfig = FlatterScatterAlloc::DefaultHeapConfig<>,
        typename T_HashConfig = FlatterScatterAlloc::DefaultFlatterScatterHashConfig,
        typename T_AlignmentPolicy = void>
    struct FlatterScatter
    {
        template<typename T>
        using AlignmentAwarePolicy = FlatterScatterImpl<T_HeapConfig, T_HashConfig, T>;

        struct Properties
        {
            using HeapConfig = T_HeapConfig;
            using HashConfig = T_HashConfig;
        };
    };


} // namespace mallocMC::CreationPolicies
