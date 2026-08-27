/* Copyright 2022 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Config.hpp"

#include <new>

namespace alpaka::core
{
    ALPAKA_FN_INLINE ALPAKA_FN_HOST auto alignedAlloc(size_t alignment, size_t size) -> void*
    {
        if(size == 0)
        {
            return nullptr;
        }
        else
        {
            return ::operator new(size, std::align_val_t{alignment});
        }
    }

    ALPAKA_FN_INLINE ALPAKA_FN_HOST void alignedFree(size_t alignment, void* ptr)
    {
        ::operator delete(ptr, std::align_val_t{alignment});
    }
} // namespace alpaka::core
