/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz

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

#include <alpaka/acc/AccCpuSerial.hpp>
#include <alpaka/atomic/AtomicAtomicRef.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/intrinsic/IntrinsicFallback.hpp>
#include <alpaka/mem/fence/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

// This is very hacky: AccCpuSerial (and in general all Accellerators) are very reluctant to be instantiated, so we do
// it the oldschool way and simply malloc some memory pretending to be that accellerator. Let's hope that null-ing it
// is a valid initialisation. The final class only has one mutable data member, so that's probably not half bad but I
// didn't go through all those hundreds of base classes. Usually, we only need the time anyways.
inline auto constructAcc()
{
    using Acc = alpaka::AccCpuSerial<alpaka::DimInt<1U>, size_t>;
    void* myPointer = malloc(sizeof(Acc));
    memset(myPointer, 0U, sizeof(Acc));
    return static_cast<Acc*>(myPointer);
}

//
static inline auto const accPointer = constructAcc();
static inline auto const& accSerial = *accPointer;

template<uint32_t T_blockSize, uint32_t T_pageSize, uint32_t T_wasteFactor = 1U, bool T_resetfreedpages = true>
struct HeapConfig
{
    static constexpr auto const accessblocksize = T_blockSize;
    static constexpr auto const pagesize = T_pageSize;
    static constexpr auto const wastefactor = T_wasteFactor;
    static constexpr auto const resetfreedpages = T_resetfreedpages;

    ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto isInAllowedRange(
        auto const& /*acc*/,
        uint32_t const chunkSize,
        uint32_t const numBytes)
    {
        return (chunkSize >= numBytes && chunkSize <= T_wasteFactor * numBytes);
    }
};

struct AlignmentPolicy
{
    struct Properties
    {
        static constexpr uint32_t const dataAlignment = 1U;
    };
};
