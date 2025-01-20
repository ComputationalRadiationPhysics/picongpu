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
#include "mallocMC/mallocMC_utils.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace mallocMC::CreationPolicies::FlatterScatterAlloc
{
    /**
     * @class PageInterpretation
     * @brief Represent our interpretation of a raw data page.
     *
     * This class takes a reference to a raw data page and a chunk size and provides an interface to this raw memory to
     * use is as a data page filled with chunks and corresponding bit masks indicating their filling. It furthermore
     * provides static helper functions that implement formulae not tied to a particular piece of memory like the
     * number of chunks given a chunk sizes (and the implicit page size).
     *
     * @param data Raw data page reference.
     * @param chunkSize Chunk sizes to interpret this memory with.
     */
    template<uint32_t T_pageSize, uint32_t T_minimalChunkSize = 1U>
    struct PageInterpretation
    {
    private:
        DataPage<T_pageSize>& data;
        uint32_t const chunkSize;

    public:
        ALPAKA_FN_INLINE ALPAKA_FN_ACC PageInterpretation(DataPage<T_pageSize>& givenData, uint32_t givenChunkSize)
            : data(givenData)
            , chunkSize(givenChunkSize)
        {
        }

        /**
         * @brief Compute the number of chunks of the given size that would fit onto a page.
         *
         * This is not quite a trivial calculation because we have to take into account the size of the bit field at
         * the end which itself depends on the number of chunks. Due to the quantisation into fixed-size bit masks we
         * are in the realm of integer divisions and remainders here.
         *
         * This is a static version of the algorithm because there's no reference to the data at all. Convenience
         * version of that uses the chunk size of an instance is provided below.
         *
         * @param chunkSize The chunk size to use for the calculation.
         * @return Number of chunks that would fit on a page.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto numChunks(uint32_t const chunkSize) -> uint32_t
        {
            constexpr auto b = static_cast<BitMaskStorageType<>>(sizeof(BitMask));
            auto const numFull = T_pageSize / (BitMaskSize * chunkSize + b);
            auto const leftOverSpace = T_pageSize - numFull * (BitMaskSize * chunkSize + b);
            auto const numInRemainder = leftOverSpace > b ? (leftOverSpace - b) / chunkSize : 0U;
            return numFull * BitMaskSize + numInRemainder;
        }

        /**
         * @brief Convenience method calling numChunks(chunkSize) with the currently set chunkSize. See there for
         * details.
         *
         * @return Number of chunks that fit on this page.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto numChunks() const -> uint32_t
        {
            return numChunks(chunkSize);
        }

        /**
         * @brief Convert a chunk index into a pointer to that piece of memory.
         *
         * @param index Chunk index < numChunks().
         * @return Pointer to that chunk.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto chunkPointer(uint32_t index) const -> void*
        {
            return reinterpret_cast<void*>(&data.data[index * chunkSize]);
        }

        /**
         * @brief Lightweight mangling of the hash into a start point for searching in the bit field.
         *
         * It is important to stress that this returns an index of a bit mask, not an individual bit's index. So, if
         * the BitMaskSize is 32 and I have 64 chunks on the page, there are two bit masks and the return value is
         * either 0 or 1, i.e. the search would start at the 0th or 32nd bit.
         *
         * @param hashValue Number providing some entropy for scattering memory accesses.
         * @return Index of a bit mask to start searching at.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto startBitMaskIndex(uint32_t const hashValue) const
        {
            return (hashValue >> 16);
        }

        /**
         * @brief Main allocation algorithm searching a free bit in the bit mask and returning the corresponding
         * pointer to a chunk.
         *
         * @param hashValue Number providing some entropy for scattering memory accesses.
         * @return Pointer to a valid piece of memory or nullptr if none was found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto create(TAcc const& acc, uint32_t const hashValue = 0U) -> void*
        {
            auto field = bitField();
            auto const index = field.firstFreeBit(acc, numChunks(), startBitMaskIndex(hashValue));
            return (index < field.noFreeBitFound()) ? chunkPointer(index) : nullptr;
        }

        /**
         * @brief Counterpart to create, freeing an allocated pointer's memory.
         *
         * In production, this does not check the validity of the pointer and providing an invalid pointer is undefined
         * behaviour. This includes valid pointers to outside the range of this page, obviously.
         *
         * @param pointer Pointer to a piece of memory created from the create method.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto destroy(TAcc const& acc, void* pointer) -> void
        {
            if(chunkSize == 0)
            {
#if(!defined(NDEBUG) && !BOOST_LANG_CUDA && !BOOST_LANG_HIP)
                throw std::runtime_error{
                    "Attempted to destroy a pointer with chunkSize==0. Likely this page was recently "
                    "(and potentially pre-maturely) freed."};
#endif // NDEBUG
                return;
            }
            auto chunkIndex = chunkNumberOf(pointer);
#if(!defined(NDEBUG) && !BOOST_LANG_CUDA && !BOOST_LANG_HIP)
            if(not isValid(acc, chunkIndex))
            {
                throw std::runtime_error{"Attempted to destroy an invalid pointer! Either the pointer does not point "
                                         "to a valid chunk or it is not marked as allocated."};
            }
#endif // NDEBUG
            bitField().unset(acc, chunkIndex);
        }

        /**
         * @brief Convenience method to retrieve the configured minimal chunk size.
         *
         * @return Minimal possible chunk size of the page.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto minimalChunkSize() -> uint32_t
        {
            return T_minimalChunkSize;
        }

        /**
         * @brief Clean up the full bit field region.
         *
         * This method is supposed to be used on raw memory and cleans up the maximal possible bit field region without
         * assuming anything about its previous content. It is supposed to be used during initialisation of raw memory
         * and after leaving a page in multi-page mode when arbitrary data is potentially found in that region. There
         * is a further optimised version of clean-up for cases where this page was in use in chunked mode before.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto cleanupFull() -> void
        {
            PageInterpretation<T_pageSize>(data, minimalChunkSize()).resetBitField();
        }

        /**
         * @brief Clean up previously unused parts of the bit field region.
         *
         * This method is supposed to have the same effect as cleanupFull but only on pages that are already in use in
         * chunked mode. Due to this additional assumption we can conclude that the part that currently acts as bit
         * field is already nulled (because we're the last ones on the page about to clean up, so all bits are unset).
         * This significantly reduces the size of the region that needs cleaning if a small chunk size was set
         * previously.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto cleanupUnused() -> void
        {
            auto worstCasePage = PageInterpretation<T_pageSize>(data, minimalChunkSize());
            memset(
                static_cast<void*>(worstCasePage.bitFieldStart()),
                0U,
                worstCasePage.bitFieldSize() - bitFieldSize());
        }

        /**
         * @brief Reset the currently used bit field to 0.
         *
         * This was introduced to be called on pages interpreted with the minimal chunk size to fully clean up the bit
         * field region.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto resetBitField() -> void
        {
            // This method is not thread-safe by itself. But it is supposed to be called after acquiring a "lock" in
            // the form of setting the filling level, so that's fine.

            memset(static_cast<void*>(bitFieldStart()), 0U, bitFieldSize());
        }

        /**
         * @brief Checks if a pointer points to an allocated chunk of memory on this page.
         *
         * This is not used in production and is not thread-safe in the sense that the information is stale as soon as
         * it's returned. It is used in debug mode and can be used for (single-threaded) tests.
         *
         * @param pointer The pointer in question.
         * @return true if the pointer points to an allocated chunk of memory, false otherwise
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isValid(TAcc const& acc, void* const pointer) const -> bool
        {
            // This function is neither thread-safe nor particularly performant. It is supposed to be used in tests and
            // debug mode.
            return isValid(acc, chunkNumberOf(pointer));
        }

    private:
        /**
         * @brief Helper method for isValid(pointer) that acts on the level of the chunk's index which translates to
         * the bit field position easier than the pointer.
         *
         * @param chunkIndex Index to a chunk to check.
         * @return true if the chunk with this index is allocated, false otherwise
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isValid(TAcc const& acc, int32_t const chunkIndex) const -> bool
        {
            return chunkIndex >= 0 and chunkIndex < static_cast<int32_t>(numChunks()) and isAllocated(acc, chunkIndex);
        }

        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto isAllocated(TAcc const& acc, uint32_t const chunkIndex) const -> bool
        {
            return bitField().get(acc, chunkIndex);
        }

    public:
        /**
         * @brief Return the bit field of this page.
         *
         * @return Bit field of this page.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto bitField() const -> BitFieldFlat
        {
            return BitFieldFlat{{bitFieldStart(), ceilingDivision(numChunks(), BitMaskSize)}};
        }

        /**
         * @brief Return a pointer to the first bit mask.
         *
         * @return Pointer to the first bit mask.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto bitFieldStart() const -> BitMask*
        {
            return reinterpret_cast<BitMask*>(&data.data[T_pageSize - bitFieldSize()]);
        }

        /**
         * @brief Convenience method to compute the bit field size of the current page. Forwards to its static version.
         * See there for details.
         *
         * @return Size of this pages bit field in number of bytes.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto bitFieldSize() const -> uint32_t
        {
            return bitFieldSize(chunkSize);
        }

        /**
         * @brief Compute the size of the bit field region in number of bytes for a page with the given chunk size.
         *
         * There is an instance method using the instance's chunk size for convenience.
         *
         * @param chunkSize Chunk size of the would-be page.
         * @return Size of this pages bit field in number of bytes.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto bitFieldSize(uint32_t const chunkSize) -> uint32_t
        {
            return sizeof(BitMask) * ceilingDivision(numChunks(chunkSize), BitMaskSize);
        }

        /**
         * @brief Commpute the maximal possible size of the bit field in number of bytes.
         *
         * This is practically the bit field size of an instance with the minimaalChunkSize().
         *
         * @return Maximal possible size of the bit field in number of bytes.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto maxBitFieldSize() -> uint32_t
        {
            return PageInterpretation<T_pageSize>::bitFieldSize(minimalChunkSize());
        }

        /**
         * @brief Compute a chunk index given a pointer.
         *
         * Please note that this will return invalid indices for invalid input pointers. Be sure to guard against this
         * if you don't want to risk messing up your memory.
         *
         * @param pointer A pointer interpreted to be pointing to a chunk of the current page.
         * @return A valid index to a chunk on this page if the pointer was valid. A potentially negative number
         * outside the valid range of chunk indices otherwise.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto chunkNumberOf(void* pointer) const -> int32_t
        {
            return indexOf(pointer, &data, chunkSize);
        }

        // these are supposed to be temporary objects, don't start messing around with them:
        PageInterpretation(PageInterpretation const&) = delete;
        PageInterpretation(PageInterpretation&&) = delete;
        auto operator=(PageInterpretation const&) -> PageInterpretation& = delete;
        auto operator=(PageInterpretation&&) -> PageInterpretation& = delete;
        ~PageInterpretation() = default;
    };
} // namespace mallocMC::CreationPolicies::FlatterScatterAlloc
