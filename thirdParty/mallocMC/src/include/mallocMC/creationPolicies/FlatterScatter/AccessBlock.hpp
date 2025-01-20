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

#include "mallocMC/creationPolicies/FlatterScatter/BitField.hpp"
#include "mallocMC/creationPolicies/FlatterScatter/DataPage.hpp"
#include "mallocMC/creationPolicies/FlatterScatter/PageInterpretation.hpp"
#include "mallocMC/mallocMC_utils.hpp"

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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

namespace mallocMC::CreationPolicies::FlatterScatterAlloc
{

    /**
     * @class PageTable
     * @brief Storage for AccessBlock's metadata
     */
    template<uint32_t T_numPages>
    struct PageTable
    {
        uint32_t chunkSizes[T_numPages]{};
        uint32_t fillingLevels[T_numPages]{};

        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto cleanup() -> void
        {
            std::fill(std::begin(chunkSizes), std::end(chunkSizes), 0U);
            std::fill(std::begin(fillingLevels), std::end(fillingLevels), 0U);
        }
    };

    /**
     * @class Padding
     * @brief Empty memory to pad the AccessBlock to the correct size
     */
    template<uint32_t T_size>
    struct Padding
    {
        char padding[T_size]{};
    };

    /**
     * @brief The C++ standard disallows zero-size arrays, so we specialise for this case.
     */
    template<>
    struct Padding<0U>
    {
    };

    /**
     * @class AccessBlock
     * @brief Coarsest memory division unit containing fixed-size pages of raw memory and metadata about their chunk
     * size and filling level
     *
     * @tparam T_HeapConfig A struct with compile-time information about the setup
     * @tparam T_AlignmentPolicy The alignment policy in use for optimisation purposes
     */
    template<typename T_HeapConfig, typename T_AlignmentPolicy>
    class AccessBlock
    {
    protected:
        static constexpr uint32_t const blockSize = T_HeapConfig::accessblocksize;
        static constexpr uint32_t const pageSize = T_HeapConfig::pagesize;
        static constexpr uint32_t const wasteFactor = T_HeapConfig::wastefactor;
        static constexpr bool const resetfreedpages = T_HeapConfig::resetfreedpages;

        using MyPageInterpretation = PageInterpretation<pageSize, T_AlignmentPolicy::Properties::dataAlignment>;

        // This class is supposed to be reinterpeted on a piece of raw memory and not instantiated directly. We set it
        // protected, so we can still test stuff in the future easily.
        AccessBlock()
        {
            init();
        }

    public:
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto init() -> void
        {
            pageTable.cleanup();
            constexpr uint32_t dummyChunkSize = 1U;
            for(auto& page : pages)
            {
                MyPageInterpretation(page, dummyChunkSize).cleanupFull();
            }
        }

        /**
         * @brief Compute the number of pages in the access block taking into account the space needed for metadata.
         *
         * @return The number of pages in the access block.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto numPages() -> uint32_t
        {
            constexpr auto numberOfPages = blockSize / (pageSize + sizeof(PageTable<1>));
            // check that the page table entries does not have a padding
            static_assert(sizeof(PageTable<numberOfPages>) == numberOfPages * sizeof(PageTable<1>));
            return numberOfPages;
        }

        /**
         * @brief Answers the question: How many successful allocations with the given size are still possible?
         * CAUTION: Not thread-safe!
         *
         * This method looks up the metadata for all its pages and computes the number of available slots with the
         * given chunk size. By doing so, the information this method is queried for is inherently not thread-safe
         * because if other threads are (de-)allocating memory during this look up, the information about each
         * individual page will be stale as soon as it is retrieved. However, beyond this inherent non-thread-safety we
         * made no effort so far to leverage parallelism or make it use atomics, i.e., move into the direction of
         * consistency in the multi-threaded case. It is supposed to run in a single thread without any interference.
         *
         * @param chunkSize The number of bytes the would-be allocations request
         * @return The number of available slots with this chunk size.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableSlots(auto const& acc, uint32_t const chunkSize) const
            -> uint32_t
        {
            if(chunkSize < multiPageThreshold())
            {
                return getAvailableChunks(acc, chunkSize);
            }
            return getAvailableMultiPages(acc, chunkSize);
        }

        /**
         * @brief Compute the index of the page a pointer points to.
         *
         * @param pointer Memory location inside of the data part of this access block.
         * @return The index of the page this pointer points to.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto pageIndex(void* pointer) const -> int32_t
        {
            return indexOf(pointer, pages, pageSize);
        }

        /**
         * @brief Verifies that a pointer points to a valid piece of memory. CAUTION: Not thread-safe!
         *
         * This method checks if a pointer is valid, meaning that it points to a chunk of memory that is marked as
         * allocated. The information it provides is inherently not thread-safe because if other threads are operating
         * on the memory, the retrieved information is stale the moment it was looked up. It is, however, consistent in
         * that it uses atomics to retrieve this information, so if the pointer is valid and does not get destroyed
         * between looking up the answer and using it (for example in the scenario where I'm the only one knowing about
         * this pointer), the answer is valid.
         *
         * @param pointer Pointer to validate
         * @return true if the pointer is valid else false
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isValid(TAcc const& acc, void* const pointer) -> bool
        {
            if(pointer == nullptr)
            {
                return false;
            }
            auto const index = pageIndex(pointer);
            auto chunkSize = atomicLoad(acc, pageTable.chunkSizes[index]);
            if(chunkSize >= pageSize)
            {
                return true;
            }
            return chunkSize == 0U or atomicLoad(acc, pageTable.fillingLevels[index]) == 0U
                       ? false
                       : interpret(index, chunkSize).isValid(acc, pointer);
        }

        /**
         * @brief Allocate a piece of memory of the given size.
         *
         * This method attempts to allocate a piece of memory of (at least) numBytes bytes. The actual size might be
         * larger (depending on the user-provided compile-time configuration of the AccessBlock) but is not
         * communicated, so it is not allowed to access the pointer outside the requested range. It returns a nullptr
         * if there is no memory available. The hashValue is used to scatter memory accesses. A cheap operation will be
         * performed to transform it into a page index to start the search at. It is also handed to the lower levels to
         * be used similarly. Having it default to 0 makes it easier for testing. The effect of this method is reverted
         * by the destroy method.
         *
         * @param numBytes Required size of memory in bytes
         * @param hashValue Optional number to scatter memory access.
         * @return A pointer to an allocated piece of memory or nullptr if no memory is available
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto create(
            TAcc const& acc,
            uint32_t const numBytes,
            uint32_t const hashValue = 0U) -> void*
        {
            void* pointer{nullptr};
            if(numBytes >= multiPageThreshold())
            {
                pointer = createOverMultiplePages(acc, numBytes, hashValue);
            }
            else
            {
                pointer = createChunk(acc, numBytes, hashValue);
            }
            return pointer;
        }

        /**
         * @brief Free up the memory a valid pointer points to.
         *
         * This method attempts to destroy the memory of a valid pointer created by the create method. It reverses the
         * effect of the create method and makes the allocated memory available for re-allocation. After calling this
         * method on a pointer it is invalid and may no longer be used for memory access. Invalid pointers are ignored
         * and a failure of this method is not communicated in production. In debug mode various exceptions can be
         * thrown for different forms of invalid pointers.
         *
         * @param pointer A pointer created by the create method.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto destroy(TAcc const& acc, void* const pointer) -> void
        {
            auto const index = pageIndex(pointer);
            if(index >= static_cast<int32_t>(numPages()) || index < 0)
            {
#if(!defined(NDEBUG) && !BOOST_LANG_CUDA && !BOOST_LANG_HIP)
                throw std::runtime_error{
                    "Attempted to destroy an invalid pointer! Pointer does not point to any page."};
#endif // NDEBUG
                return;
            }
            auto const chunkSize = atomicLoad(acc, pageTable.chunkSizes[index]);
            if(chunkSize >= multiPageThreshold())
            {
                destroyOverMultiplePages(acc, index, chunkSize);
            }
            else
            {
                destroyChunk(acc, pointer, index, chunkSize);
            }
        }

    private:
        DataPage<pageSize> pages[numPages()]{};
        PageTable<numPages()> pageTable{};
        Padding<blockSize - sizeof(pages) - sizeof(pageTable)> padding{};

        /**
         * @brief The number of bytes at which allocation switch to "multi-page mode", i.e., allocate full pages.
         *
         * It is obvious that this number can be at most page size subtracted by the size of one bit mask. There is,
         * however, no strict lower bound because we theoretically disregard the lower levels completely by this
         * switch. If we reasonably assume that our lower hierarchy levels add value (i.e. performance) to our
         * implementation, a reasonable lower bound would be the size at which only a single allocation fits onto a
         * page. This method could be used for fine-tuning performance in that sense.
         *
         * @return The number of bytes at which to switch to multi-page mode.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto multiPageThreshold() -> uint32_t
        {
            return ceilingDivision(pageSize - sizeof(BitMaskStorageType<>), 2U);
        }

        /**
         * @brief Convenience method that creates a PageInterpretation from a page identified by its page index and a
         * chunk size.
         *
         * @param pageIndex Identifies the page in the array of raw pages.
         * @param chunkSize Chunk size for which to interpret the page.
         * @return A page interpretation of the requested page.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto interpret(uint32_t const pageIndex, uint32_t const chunkSize)
        {
            return MyPageInterpretation(pages[pageIndex], chunkSize);
        }

        /**
         * @brief Branch of getAvailableSlots for chunk sizes below the multi-page threshold. See there for details.
         *
         * @param chunkSize Would-be allocation size to test for.
         * @return Number of allocations that would succeed with this size.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableChunks(auto const& acc, uint32_t const chunkSize) const
            -> uint32_t
        {
            // TODO(lenz): This is not thread-safe!
            return std::transform_reduce(
                std::cbegin(pageTable.chunkSizes),
                std::cend(pageTable.chunkSizes),
                std::cbegin(pageTable.fillingLevels),
                0U,
                std::plus<uint32_t>{},
                [this, &acc, chunkSize](auto const localChunkSize, auto const fillingLevel)
                {
                    auto const numChunks
                        = MyPageInterpretation::numChunks(localChunkSize == 0 ? chunkSize : localChunkSize);
                    return ((this->isInAllowedRange(acc, localChunkSize, chunkSize) or localChunkSize == 0U)
                            and fillingLevel < numChunks)
                               ? numChunks - fillingLevel
                               : 0U;
                });
        }

        /**
         * @brief Branch of getAvailableSlots for chunk sizes above the multi-page threshold. See there for details.
         *
         * @param chunkSize Would-be allocation size to test for.
         * @return Number of allocations that would succeed with this size.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getAvailableMultiPages(auto const& /*acc*/, uint32_t const chunkSize) const
            -> uint32_t
        {
            // TODO(lenz): This is not thread-safe!
            auto numPagesNeeded = ceilingDivision(chunkSize, pageSize);
            if(numPagesNeeded > numPages())
            {
                return 0U;
            }
            uint32_t sum = 0U;
            for(uint32_t i = 0; i < numPages() - numPagesNeeded + 1;)
            {
                if(std::all_of(
                       pageTable.chunkSizes + i,
                       pageTable.chunkSizes + i + numPagesNeeded,
                       [](auto const& val) { return val == 0U; }))
                {
                    sum += 1;
                    i += numPagesNeeded;
                }
                else
                {
                    ++i;
                }
            }
            return sum;
        }

        /**
         * @brief Creation algorithm in multi-page mode.
         *
         * In this mode, we have decided to ignore all the lower level hierarchy. The algorithm simplifies accordingly
         * and a few optimisations can be done. It can however be quite cumbersome to find a sufficient number of
         * contiguous pages, so this will likely be most performant for small sizes.
         *
         * @param numBytes Required allocation size in number of bytes.
         * @param hashValue A hash value used to scatter memory access.
         * @return Pointer to a valid piece of memory or nullptr if no such memory was found.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto createOverMultiplePages(
            auto const& acc,
            uint32_t const numBytes,
            uint32_t hashValue) -> void*
        {
            auto numPagesNeeded = ceilingDivision(numBytes, +pageSize);
            if(numPagesNeeded > numPages())
            {
                return static_cast<void*>(nullptr);
            }

            // We take a little head start compared to the chunked case in order to not have them interfere with our
            // laborious search for contiguous pages.
            auto startIndex = startPageIndex(acc, hashValue) + numPagesNeeded;
            return wrappingLoop(
                acc,
                startIndex,
                numPages() - (numPagesNeeded - 1),
                static_cast<void*>(nullptr),
                [&](auto const& localAcc, auto const& firstIndex)
                {
                    void* result{nullptr};
                    auto numPagesAcquired = acquirePages(localAcc, firstIndex, numPagesNeeded);
                    if(numPagesAcquired == numPagesNeeded)
                    {
                        // At this point, we have acquired all the pages we need and nobody can mess with them anymore.
                        // We still have to set the chunk size correctly.
                        setChunkSizes(localAcc, firstIndex, numPagesNeeded, numBytes);
                        result = &pages[firstIndex];
                    }
                    else
                    {
                        releasePages(localAcc, firstIndex, numPagesAcquired);
                    }
                    return result;
                });
        }

        /**
         * @brief Short-circuiting acquisition of multiple contiguous pages.
         *
         * The algorithm attempts to acquire the requested number of pages starting from firstIndex locking them by
         * setting their filling level to page size. It returns when either all requested pages are acquired or an
         * already occupied page was hit. In either case, it returns the number of successful acquisitions. This method
         * does not clean up after itself, i.e., it does not release the pages in case of failure.
         *
         * @param firstIndex Start index of the array of contiguous pages.
         * @param numPagesNeeded Number of pages to be acquired.
         * @return Number of pages that were successfully acquired. This is smaller than numPagesNeeded if the method
         * failed.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto acquirePages(
            auto const& acc,
            uint32_t const firstIndex,
            uint32_t const numPagesNeeded) -> uint32_t
        {
            uint32_t index = 0U;
            uint32_t oldFilling = 0U;
            for(index = 0U; index < numPagesNeeded; ++index)
            {
                oldFilling = alpaka::atomicCas(acc, &pageTable.fillingLevels[firstIndex + index], 0U, +pageSize);
                if(oldFilling != 0U)
                {
                    break;
                }
            }
            return index;
        }

        /**
         * @brief Counterpart to acquirePages for doing the clean-up in case of failure.
         *
         * This method starts from page firstIndex and releases the lock of numPagesAcquired contiguous pages. This is
         * supposed to be called in the case of failure of acquirePages to release the already acquired pages.
         *
         * @param firstIndex Start index of the array of contiguous pages.
         * @param numPagesAcquired Number of pages to be released.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto releasePages(
            auto const& acc,
            uint32_t const firstIndex,
            uint32_t const numPagesAcquired) -> void
        {
            for(uint32_t index = 0U; index < numPagesAcquired; ++index)
            {
                alpaka::atomicSub(acc, &pageTable.fillingLevels[firstIndex + index], +pageSize);
            }
        }

        /**
         * @brief Set the chunk sizes of a contiguous array of pages.
         *
         * This function assumes that all the pages are locked by the current thread and performs a hard set operation
         * without checking the previous content.
         *
         * @param firstIndex Start index of the contiguous array of pages.
         * @param numPagesNeeded The number of pages to set the chunk size on.
         * @param numBytes Chunk size to be set in number of bytes.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto setChunkSizes(
            auto const& acc,
            uint32_t const firstIndex,
            uint32_t const numPagesNeeded,
            uint32_t const numBytes) -> void
        {
            for(uint32_t numPagesAcquired = 0U; numPagesAcquired < numPagesNeeded; ++numPagesAcquired)
            {
                // At this point in the code, we have already locked all the pages. So, we literally don't care what
                // other threads thought this chunk size would be because we are the only ones legitimately messing
                // with this page. This chunk size may be non-zero because we could have taken over a page before it
                // was properly cleaned up. That is okay for us because we're handing out uninitialised memory anyways.
                // But it is very important to record the correct chunk size here, so the destroy method later on knows
                // how to handle this memory.
                alpaka::atomicExch(acc, &pageTable.chunkSizes[firstIndex + numPagesAcquired], numBytes);
            }
        }

        /**
         * @brief Special return value for an unsuccessful search of available pages.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto noFreePageFound()
        {
            return numPages();
        }

        /**
         * @brief Compute an index where to start searching for a free page from a hash value.
         *
         * @param hashValue Hash value to introduce some entropy here.
         * @return Start index for searching a free page.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto startPageIndex(auto const& /*acc*/, uint32_t const hashValue)
        {
            return (hashValue >> 8U) % numPages();
        }

        /**
         * @brief Helper that combines the necessary checks to ensure a page index is valid.
         *
         * @param index The page index to check.
         * @return true if the page index is valid and false otherwise
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isValidPageIdx(uint32_t const index) const -> bool
        {
            return index != noFreePageFound() && index < numPages();
        }

        /**
         * @brief Main algorithm to create a chunk of memory on a page.
         *
         * This is the main algorithm for creating a chunk of memory. It searches for a free page and instructs it to
         * create some memory. If successful, it returns this pointer. If not, it searches on.
         *
         * @param numBytes Number of bytes required.
         * @param hashValue A hash value used to scatter the memory accesses.
         * @return A pointer to a valid piece of memory or nullptr if no available memory could be found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto createChunk(
            TAcc const& acc,
            uint32_t const numBytes,
            uint32_t const hashValue) -> void*
        {
            auto index = startPageIndex(acc, hashValue);

            // Under high pressure, this loop could potentially run for a long time because the information where and
            // when we started our search is not maintained and/or used. This is a feature, not a bug: Given a
            // consistent state, the loop will terminate once a free chunk is found or when all chunks are filled for
            // long enough that `choosePage` could verify that each page is filled in a single run.
            //
            // The seemingly non-terminating behaviour that we wrap around multiple times can only occur (assuming a
            // consistent, valid state of the data) when there is high demand for memory such that pages that appear
            // free to `choosePage` are repeatedly found but then the free chunks are scooped away by other threads.
            //
            // In the latter case, it is considered desirable to wrap around multiple times until the thread was fast
            // enough to acquire some memory.
            void* pointer = nullptr;
            do
            {
                // TODO(lenz): This can probably be index++.
                index = (index + 1) % numPages();
                uint32_t chunkSize = numBytes;
                index = choosePage(acc, numBytes, index, chunkSize);
                if(isValidPageIdx(index))
                {
                    pointer = MyPageInterpretation{pages[index], chunkSize}.create(acc, hashValue);
                    if(pointer == nullptr)
                    {
                        leavePage(acc, index);
                    }
                }
            } while(isValidPageIdx(index) and pointer == nullptr);
            return pointer;
        }

        /**
         * @brief Main loop running over all pages checking for available ones.
         *
         * It is important to stress that the information about availability of the returned page is already stale when
         * it is returned. Thus, it can well happen that an actual allocation attempt on this page still fails, e.g.,
         * because another thread was faster and scooped away that piece of memory.
         *
         * @param numBytes Required allocation size in number of bytes.
         * @param startIndex Index of the page to start the search from.
         * @param chunkSizeCache A memory location to store a local copy of the current chunk size. Used for
         * optimisation by reducing the number of atomic lookups.
         * @return A page index to a potntially available page or noFreePageFound() if none was found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto choosePage(
            TAcc const& acc,
            uint32_t const numBytes,
            uint32_t const startIndex,
            uint32_t& chunkSizeCache) -> uint32_t
        {
            return wrappingLoop(
                acc,
                startIndex,
                numPages(),
                noFreePageFound(),
                [this, numBytes, &chunkSizeCache](auto const& localAcc, auto const index) {
                    return this->thisPageIsSuitable(localAcc, index, numBytes, chunkSizeCache) ? index
                                                                                               : noFreePageFound();
                });
        }

        /**
         * @brief Helper function combining checks to match the requested number of bytes with a found chunk size
         * taking into account the waste factor.
         *
         * @param chunkSize Actually found chunk sizes of a page in number of bytes
         * @param numBytes Requested allocation size in number of bytes.
         * @return
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isInAllowedRange(
            auto const& acc,
            uint32_t const chunkSize,
            uint32_t const numBytes) const
        {
            return T_HeapConfig::isInAllowedRange(acc, chunkSize, numBytes);
        }

        /**
         * @brief Checks if a page is usable for allocation of numBytes and enters it.
         *
         * This method looks up the metdata of the page identified by its index to check if we can hope for a
         * successful allocation there. In doing so, it enters the page (i.e. increments its filling level) and, if
         * necessary, already sets the correct chunk size. In a multi-threaded context the separate concerns of
         * checking and setting cannot be split because the information used for the check would already be stale at
         * the time of setting anything. If it returns true, the filling level and chunk sizes are thus suitable for
         * proceeding further and the caller is responsible for cleaning up appropriately if a failure at a later stage
         * occurs. If it returns false, it has already cleaned up everything itself and there is no further action
         * required on the caller's side.
         *
         * @param index Index to identify the page among the raw data pages.
         * @param numBytes Requested allocation size in number of bytes.
         * @param chunkSizeCache A memory location to store a local copy of the current chunk size. Used for
         * optimisation by reducing the number of atomic lookups.
         * @return true if the page is suitable and false otherwise
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto thisPageIsSuitable(
            TAcc const& acc,
            uint32_t const index,
            uint32_t const numBytes,
            uint32_t& chunkSizeCache) -> bool
        {
            bool suitable = false;
            auto oldFilling = enterPage(acc, index);

            // At this point, we're only testing against our desired `numBytes`. Due to the `wastefactor` the actual
            // `chunkSize` of the page might be larger and, thus, the actual `numChunks` might be smaller than what
            // we're testing for here. But if this fails already, we save one atomic.
            if(oldFilling < MyPageInterpretation::numChunks(numBytes))
            {
                uint32_t oldChunkSize = alpaka::atomicCas(acc, &pageTable.chunkSizes[index], 0U, numBytes);
                chunkSizeCache = oldChunkSize == 0U ? numBytes : oldChunkSize;

                // Now that we know the real chunk size of the page, we can check again if our previous assessment was
                // correct. But first we need to make sure that we are actually in chunked mode. This will be redundant
                // with the second check in most situations because we usually would choose a multi-page threshold that
                // would not switch to multi-page mode while more than one chunk fits on the page but this is a design
                // decision that could change in the future.
                if(oldChunkSize < multiPageThreshold()
                   and oldFilling < MyPageInterpretation::numChunks(chunkSizeCache))
                {
                    suitable = isInAllowedRange(acc, chunkSizeCache, numBytes);
                }
            }
            if(not suitable)
            {
                leavePage(acc, index);
            }
            return suitable;
        }

        /**
         * @brief Counterpart to createChunk freeing up a piece of memory in the chunked mode. See destroy for details.
         *
         * This is the most difficult part of the algorithm. We will successively remove our metadata from the various
         * levels and must be extra careful which information we can still rely on. Most of this complexity is captured
         * in leavePage.
         *
         * @param pointer Pointer to a valid piece of memory created by createChunk.
         * @param pageIndex Index of the page the pointer points to. Supplying this is an optimisation because it was
         * already computed on a higher level in the call stack. This information would already be contained in
         * pointer.
         * @param chunkSize Chunk size of the page we're operating on. This is potentially different from the size of
         * memory the pointer points to due to the waste factor.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void destroyChunk(
            TAcc const& acc,
            void* pointer,
            uint32_t const pageIndex,
            uint32_t const chunkSize)
        {
            auto page = interpret(pageIndex, chunkSize);
            page.destroy(acc, pointer);
            leavePage(acc, pageIndex);
        }

        /**
         * @brief Enter a page for any purpose.
         *
         * This method is very important. We maintain the invariant that any potentially writing access to a page
         * starts by entering and ends by leaving a page. These are currently implemented as updating the filling level
         * accordingly. You are not allowed to touch a page unless you have entered it (although multi-page mode uses a
         * shortcut here). This implies that we always have to check the filling level before checking for the chunk
         * size.
         *
         * @param pageIndex Identifies the page in the array of raw data pages.
         * @return The old filling level for further checks.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto enterPage(TAcc const& acc, uint32_t const pageIndex) -> uint32_t
        {
            auto const oldFilling = alpaka::atomicAdd(acc, &pageTable.fillingLevels[pageIndex], 1U);
            // We assume that this page has the correct chunk size. If not, the chunk size is either 0 (and oldFilling
            // must be 0, too) or the next check will fail.
            return oldFilling;
        }

        /**
         * @brief Leave a page after any potentially modifying operation on it.
         *
         * This method must be called whenever you have entered a page (using enterPage()). This is a very subtle and
         * error-prone method because we are successively removing metadata and need to be extra careful which
         * information and guards we can still trust. In the simplest case, there's not much to do but decrease the
         * filling level but potentially we're the last thread on the page and need to clean up remaining metadata for
         * the threads to come. In that case, we explicitly allow for threads to take over the page as-is to spare us
         * the trouble of cleaning up. But doing so opens up many subtle ways of reordering memory accesses. Also, we
         * cannot rely in much previous information (like chunk sizes looked up earlier) because other threads might
         * have already updated them. Be warned!
         *
         * @param pageIndex Identifies the page in the array of raw data pages.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void leavePage(TAcc const& acc, uint32_t const pageIndex)
        {
            // This outermost atomicSub is an optimisation: We can fast-track this if we are not responsible for the
            // clean-up. Using 0U -> 1U in the atomicCAS and comparison further down would have the same effect (if the
            // else branch contained the simple subtraction). It's a matter of which case shall have one operation
            // less.
            auto originalFilling = alpaka::atomicSub(acc, &pageTable.fillingLevels[pageIndex], 1U);

            if constexpr(resetfreedpages)
            {
                if(originalFilling == 1U)
                {
                    // CAUTION: This section has caused a lot of headaches in the past. We're in a state where the
                    // filling level is 0 but we have not properly cleaned up the page and the metadata yet. This is on
                    // purpose because another thread might still take over this page and spare us the trouble of
                    // freeing everything up properly. But this other thread must take into account the possibility
                    // that it acquired a second-hand page. Look here if you run into another deadlock. It might well
                    // be related to this section.

                    auto lock = pageSize;
                    auto latestFilling = alpaka::atomicCas(acc, &pageTable.fillingLevels[pageIndex], 0U, lock);
                    if(latestFilling == 0U)
                    {
                        auto chunkSize = atomicLoad(acc, pageTable.chunkSizes[pageIndex]);
                        if(chunkSize != 0)
                        {
                            // At this point it's guaranteed that the fiilling level is numChunks and thereby locked.
                            // Furthermore, chunkSize cannot have changed because we maintain the invariant that the
                            // filling level is always considered first, so no other thread can have passed that
                            // barrier to reset it.
                            MyPageInterpretation{pages[pageIndex], chunkSize}.cleanupUnused();
                            alpaka::mem_fence(acc, alpaka::memory_scope::Device{});

                            // It is important to keep this after the clean-up line above: Otherwise another thread
                            // with a smaller chunk size might circumvent our lock and already start allocating before
                            // we're done cleaning up.
                            alpaka::atomicCas(acc, &pageTable.chunkSizes[pageIndex], chunkSize, 0U);
                        }

                        // TODO(lenz): Original version had a thread fence at this point in order to invalidate
                        // potentially cached bit masks. Check if that's necessary!

                        // At this point, there might already be another thread (with another chunkSize) on this page
                        // but that's fine. It won't see the full capacity but we can just subtract what we've added
                        // before:
                        alpaka::atomicSub(acc, &pageTable.fillingLevels[pageIndex], lock);
                    }
                }
            }
        }

        /**
         * @brief Counterpart to createOverMultiplePages, freeing up memory in multi-page mode.
         *
         * This method is way simpler than its chunked version because in multi-page mode we have a hard lock on the
         * pages we acquired and are free to manipulate them to our will. We just make sure that releasing this lock is
         * the last operation we perform.
         *
         * @param pageIndex Identifies the first page in the array of raw data pages.
         * @param chunkSize The chunk size set on that first page (i.e. the original allocation size).
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void destroyOverMultiplePages(
            auto const& acc,
            uint32_t const pageIndex,
            uint32_t const chunkSize)
        {
            auto numPagesNeeded = ceilingDivision(chunkSize, pageSize);
            for(uint32_t i = 0; i < numPagesNeeded; ++i)
            {
                auto myIndex = pageIndex + i;
                // Everything inside the following scope is done to reset the free'd pages. As opposed to the chunked
                // case, we decided to always perform a reset in multi-page mode regardless of the value of
                // `resetfreedpages`. If you want to reinstate the old behaviour or add a second parameter
                // specifically for multi-page mode, e.g., resetreedpages_multipage, just put an `if constexpr` around
                // here again.
                {
                    MyPageInterpretation{pages[myIndex], T_AlignmentPolicy::Properties::dataAlignment}.cleanupFull();
                    alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
                    alpaka::atomicCas(acc, &pageTable.chunkSizes[myIndex], chunkSize, 0U);
                }
                alpaka::atomicSub(acc, &pageTable.fillingLevels[myIndex], +pageSize);
            }
        }
    };

} // namespace mallocMC::CreationPolicies::FlatterScatterAlloc
