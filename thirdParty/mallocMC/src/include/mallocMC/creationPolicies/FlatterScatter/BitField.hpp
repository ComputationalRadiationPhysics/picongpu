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

#include "mallocMC/creationPolicies/FlatterScatter/wrappingLoop.hpp"
#include "mallocMC/mallocMC_utils.hpp"

#include <alpaka/core/Common.hpp>
#include <alpaka/intrinsic/Traits.hpp>

#include <sys/types.h>

#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace mallocMC::CreationPolicies::FlatterScatterAlloc
{
    namespace detail
    {
        template<uint32_t size>
        struct BitMaskStorageTypes
        {
            using type = void;
        };

        template<>
        struct BitMaskStorageTypes<16U>
        {
            using type = uint16_t;
        };

        template<>
        struct BitMaskStorageTypes<32U>
        {
            using type = uint32_t;
        };

        template<>
        struct BitMaskStorageTypes<64U>
        {
            using type = uint64_t;
        };
    } // namespace detail

    /**
     * @brief Number of bits in a bit mask. Most likely you want a power of two here.
     */
    constexpr uint32_t const BitMaskSize = 32U;

    /**
     * @brief Type to store the bit masks in. It's implemented as a template in order to facilitate changing the type
     * depending on BitMaskSize. Use it with its default template argument in order to make your code agnostic of the
     * number configured in BitMaskSize. (Up to providing a template implementation, of course.)
     */
    template<uint32_t size = BitMaskSize>
    using BitMaskStorageType = detail::BitMaskStorageTypes<size>::type;

    /**
     * @brief Represents a completely filled bit mask, i.e., all bits are one.
     */
    template<uint32_t size = BitMaskSize>
    static constexpr BitMaskStorageType<size> const allOnes = std::numeric_limits<BitMaskStorageType<size>>::max();

    /**
     * @brief Return the bit mask's underlying type with a single bit set (=1) at position index and all others unset
     * (=0).
     *
     * @param index Position of the single bit set.
     * @return Bit mask's underlying type with one bit set.
     */
    template<uint32_t size = BitMaskSize>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto singleBit(uint32_t const index) -> BitMaskStorageType<size>
    {
        return BitMaskStorageType<size>{1U} << index;
    }

    /**
     * @brief Return the bit mask's underlying type with all bits up to index from the right are set (=1) and all
     * higher bits are unset (=0).
     *
     * @param index Number of set bits.
     * @return Bit mask's underlying type with index bits set.
     */
    template<typename TAcc, uint32_t size = BitMaskSize>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC auto allOnesUpTo(uint32_t const index) -> BitMaskStorageType<size>
    {
        return index == 0 ? 0 : (allOnes<size> >> (size - index));
    }

    /**
     * @class BitMaskImpl
     * @brief Represents a bit mask basically wrapping the BitMaskStorageType<>.
     *
     * This class basically provides a convenience interface to the (typically integer) type BitMaskStorageType<> for
     * bit manipulations. It was originally modelled closely after std::bitset which is not necessarily available on
     * device for all compilers, etc.
     *
     * Convention: We start counting from the right, i.e., if mask[0] == 1 and all others are 0, then mask = 0...01
     *
     * CAUTION: This convention is nowhere checked and we might have an implicit assumption on the endianess here. We
     * never investigated because all architectures we're interested in have the same endianess and it works on them.
     *
     */
    template<uint32_t MyBitMaskSize = BitMaskSize>
    struct BitMaskImpl
    {
        BitMaskStorageType<MyBitMaskSize> mask{};

        /**
         * @return An invalid bit index indicating the failure of a search in the bit mask.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto noFreeBitFound() -> uint32_t
        {
            return MyBitMaskSize;
        }

        /**
         * @brief Look up if the index-th bit is set.
         *
         * @param index Bit position to check.
         * @return true if bit is set else false.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator()(TAcc const& acc, auto const index) -> bool
        {
            return (atomicLoad(acc, mask) & singleBit<MyBitMaskSize>(index)) != BitMaskStorageType<MyBitMaskSize>{0U};
        }

        /**
         * @brief Set all bits (to 1).
         *
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto set(TAcc const& acc) -> BitMaskStorageType<MyBitMaskSize>
        {
            return alpaka::atomicOr(
                acc,
                &mask,
                static_cast<BitMaskStorageType<MyBitMaskSize>>(+allOnes<MyBitMaskSize>));
        }

        /**
         * @brief Set the index-th bit (to 1).
         *
         * @param index Bit position to set.
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto set(TAcc const& acc, auto const index)
        {
            return alpaka::atomicOr(acc, &mask, singleBit<MyBitMaskSize>(index));
        }

        /**
         * @brief Unset the index-th bit (set it to 0).
         *
         * @param index Bit position to unset.
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto unset(TAcc const& acc, auto const index)
        {
            return alpaka::atomicAnd(
                acc,
                &mask,
                static_cast<BitMaskStorageType<MyBitMaskSize>>(
                    allOnes<MyBitMaskSize> ^ singleBit<MyBitMaskSize>(index)));
        }

        /**
         * @brief Flip all bits in the mask.
         *
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto flip(TAcc const& acc)
        {
            return alpaka::atomicXor(
                acc,
                &mask,
                static_cast<BitMaskStorageType<MyBitMaskSize>>(+allOnes<MyBitMaskSize>));
        }

        /**
         * @brief Flip the index-th bits in the mask.
         *
         * @param index Bit position to flip.
         * @return Previous mask.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto flip(TAcc const& acc, auto const index)
        {
            return alpaka::atomicXor(
                acc,
                &mask,
                static_cast<BitMaskStorageType<MyBitMaskSize>>(singleBit<MyBitMaskSize>(index)));
        }

        /**
         * @brief Compare with another mask represented by a BitMaskStorageType<>. CAUTION: This does not use atomics
         * and is not thread-safe!
         *
         * This is not implemented thread-safe because to do so we'd need to add the accelerator as a function argument
         * and that would not abide by the interface for operator==. It's intended use is to make (single-threaded)
         * tests more readable, so that's not an issue.
         *
         * @param other Mask to compare with.
         * @return true if all bits are identical else false.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator==(BitMaskStorageType<MyBitMaskSize> const other) const -> bool
        {
            return (mask == other);
        }

        /**
         * @brief Spaceship operator comparing with other bit masks. CAUTION: This does not use atomics and is not
         * thread-safe! See operator== for an explanation.
         *
         * @param other Bit mask to compare with.
         * @return Positive if this mask > other mask, 0 for equality, negative otherwise.
         */
        // My version of clang cannot yet handle the spaceship operator apparently:
        // clang-format off
         ALPAKA_FN_INLINE ALPAKA_FN_ACC auto operator<=> (BitMaskImpl const other) const
        // clang-format on
        {
            return (mask - other.mask);
        }

        /**
         * @brief Check if no bit is set (=1).
         *
         * @return true if no bit is set else false.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto none() const -> bool
        {
            return mask == 0U;
        }

        /**
         * @brief Interface to the main algorithm of finding a free bit.
         *
         * This algorithm searches for an unset bit and returns its position as an index (which is supposed to be
         * translated into a corresponding chunk by the PageInterpretation). Upon doing so, it also sets this bit
         * because in a multi-threaded context we cannot separate the concerns of retrieving information and acting on
         * the information. It takes a start index that acts as an initial guess but (in the current implementation) it
         * does not implement a strict wrapping loop as the other stages do because this would waste valuable
         * information obtained from the collective operation on all bits in the mask.
         *
         * Additionally, it copes with partial masks by ignoring all bit positions beyond numValidBits.
         *
         * @param numValidBits Bit positions beyond this number will be ignored.
         * @param initialGuess Initial guess for the first look up.
         * @return Bit position of a free bit or noFreeBitFound() in the case of none found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBit(
            TAcc const& acc,
            uint32_t const numValidBits = MyBitMaskSize,
            uint32_t const initialGuess = 0) -> uint32_t
        {
            return firstFreeBitWithInitialGuess(acc, initialGuess % MyBitMaskSize, numValidBits);
        }

    private:
        /**
         * @brief Implementation of the main search algorithm. See the public firstFreeBit method for general details.
         * This version assumes a valid range of the input values.
         *
         * @param initialGuess Initial guess for the first look up must be in the range [0;MyBitMaskSize).
         * @param endIndex Maximal position to consider. Bits further out will be ignored.
         * @return Bit position of a free bit or noFreeBitFound() in the case of none found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBitWithInitialGuess(
            TAcc const& acc,
            uint32_t const initialGuess,
            uint32_t const endIndex) -> uint32_t
        {
            auto result = noFreeBitFound();
            BitMaskStorageType<MyBitMaskSize> oldMask = 0U;

            // This avoids a modulo that's not a power of two and is faster thereby.
            auto const selectedStartBit = initialGuess >= endIndex ? 0U : initialGuess;
            for(uint32_t i = selectedStartBit; i < endIndex and result == noFreeBitFound();)
            {
                oldMask = alpaka::atomicOr(acc, &mask, singleBit<MyBitMaskSize>(i));
                if((oldMask & singleBit<MyBitMaskSize>(i)) == 0U)
                {
                    result = i;
                }

                // In case of no free bit found, this will return -1. Storing it in a uint32_t will underflow and
                // result in 0xffffffff but that's okay because it also ends the loop as intended.
                i = alpaka::ffs(acc, static_cast<std::make_signed_t<BitMaskStorageType<MyBitMaskSize>>>(~oldMask)) - 1;
            }

            return result;
        }
    };

    using BitMask = BitMaskImpl<BitMaskSize>;

    /**
     * @class BitFieldFlat
     * @brief Represents a (non-owning) bit field consisting of multiple bit masks.
     *
     * This class interprets a piece of memory as an array of bit masks and provides convenience functionality to act
     * on them as a long array of bits. Most importantly, it provides an interface to find a free bit. It is a
     * non-owning view of the memory!
     *
     * Please note, that methods usually (unless stated otherwise) refer to bits counting all bits from the start of
     * the bit field, so if BitMask size is 32 and index=34=31+3, we're checking for the third bit of the second mask
     * (if masks was a matrix this would be equivalent to: masks[1][2]).
     *
     */
    template<uint32_t MyBitMaskSize = BitMaskSize>
    struct BitFieldFlatImpl
    {
        std::span<BitMaskImpl<MyBitMaskSize>> data;

        /**
         * @brief Check if the index-th bit in the bit field is set (=1).
         *
         * @param index Bit position to check.
         * @return true if bit is set else false.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto get(TAcc const& acc, uint32_t index) const -> bool
        {
            return data[index / MyBitMaskSize](acc, index % MyBitMaskSize);
        }

        /**
         * @brief Get the index-th mask NOT bit (counting in number of masks and not bits).
         *
         * @param index Position of the mask.
         * @return Requested mask.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto getMask(uint32_t const index) const -> BitMaskImpl<MyBitMaskSize>&
        {
            return data[index];
        }

        /**
         * @brief Set the index-th bit (to 1).
         *
         * @param index Position of the bit.
         * @return
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void set(TAcc const& acc, uint32_t const index) const
        {
            data[index / MyBitMaskSize].set(acc, index % MyBitMaskSize);
        }

        /**
         * @brief Counterpart to set, unsetting (to 0) to index-th bit.
         *
         * @tparam TAcc
         * @param acc
         * @param index
         * @return
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC void unset(TAcc const& acc, uint32_t const index) const
        {
            data[index / MyBitMaskSize].unset(acc, index % MyBitMaskSize);
        }

        /**
         * @return Begin iterator to the start of the array of masks, iterating over masks NOT bits.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto begin() const
        {
            return std::begin(data);
        }

        /**
         * @return End iterator to the start of the array of masks, iterating over masks NOT bits.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto end() const
        {
            return std::end(data);
        }

        /**
         * @brief Count the number of masks.
         *
         * @return Number of masks.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto numMasks() const
        {
            return data.size();
        }

        /**
         * @brief Count the number of bits in the array of masks.
         *
         * This does not take into account if bits are valid or not, so this is always a multiple of the MyBitMaskSize
         * currently.
         *
         * @return Number of bits.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto numBits() const
        {
            return numMasks() * MyBitMaskSize;
        }

        /**
         * @brief Main algorithm for finding and setting a free bit in the bit field.
         *
         * This iterates through the masks wrapping around from the given startIndex. The information of how many bits
         * are valid is passed through the lower levels which automatically discard out of range results (accounting of
         * partially filled masks). As always, we can't separate the concerns of retrieving information and acting on
         * it in a multi-threaded context, so if a free bit is found it is immediately set.
         *
         * @param numValidBits Number of valid bits in the bit field (NOT masks, i.e. it's equal to numChunks() on the
         * page). Should typically be a number from the range [MyBitMaskSize * (numMasks()-1) + 1, MyBitMaskSize *
         * numMasks()) although other numbers shouldn't hurt.
         * @param startIndex Bit mask (NOT bit) to start the search at.
         * @return The index of the free bit found (and set) or noFreeBitFound() if none was found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBit(
            TAcc const& acc,
            uint32_t numValidBits,
            uint32_t const startIndex = 0U) -> uint32_t
        {
            return wrappingLoop(
                acc,
                startIndex % numMasks(),
                numMasks(),
                noFreeBitFound(),
                [this, numValidBits](TAcc const& localAcc, auto const index)
                {
                    auto tmp = this->firstFreeBitAt(localAcc, numValidBits, index);
                    return tmp;
                });
        }

        /**
         * @return Special invalid bit index to indicate that no free bit was found.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto noFreeBitFound() const -> uint32_t
        {
            return numBits();
        }

    private:
        /**
         * @return Position inside of a mask to start the search at.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto startBitIndex()
        {
            return laneid();
        }

        /**
         * @brief Helper function checking if we're in the last mask.
         *
         * @param numValidBits Number of valid bits in the bit field. The mask containing this bit is the last mask.
         * @param maskIndex Index of the mask under consideration (NOT bit).
         * @return true if the mask is the last mask else false.
         */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto isThisLastMask(
            uint32_t const numValidBits,
            uint32_t const maskIndex)
        {
            // >= in case index == numValidBits - MyBitMaskSize
            return (maskIndex + 1) * MyBitMaskSize >= numValidBits;
        }

        /**
         * @brief Implementation of the main algorithm asking a mask of a free bit and checking if the answer is valid.
         *
         * @param numValidBits Number of valid bits in the bit field.
         * @param maskIdx Index of the maks under consideration.
         * @return Index of the free bit found IN THE BITFIELD (not only in the mask, so this value can be larger than
         * MyBitMaskSize) or noFreeBitFound() if none was found.
         */
        template<typename TAcc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC auto firstFreeBitAt(
            TAcc const& acc,
            uint32_t const numValidBits,
            uint32_t const maskIdx) -> uint32_t
        {
            auto numValidBitsInLastMask = (numValidBits ? ((numValidBits - 1U) % MyBitMaskSize + 1U) : 0U);
            auto indexInMask = getMask(maskIdx).firstFreeBit(
                acc,
                isThisLastMask(numValidBits, maskIdx) ? numValidBitsInLastMask : MyBitMaskSize,
                startBitIndex());
            if(indexInMask < BitMaskImpl<MyBitMaskSize>::noFreeBitFound())
            {
                uint32_t freeBitIndex = indexInMask + MyBitMaskSize * maskIdx;
                if(freeBitIndex < numValidBits)
                {
                    return freeBitIndex;
                }
            }
            return noFreeBitFound();
        }
    };

    using BitFieldFlat = BitFieldFlatImpl<BitMaskSize>;
} // namespace mallocMC::CreationPolicies::FlatterScatterAlloc
