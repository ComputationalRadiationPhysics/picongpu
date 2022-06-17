/* Copyright 2022 René Widera, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>

#include <new>

namespace alpaka::core
{
    ALPAKA_FN_INLINE ALPAKA_FN_HOST auto alignedAlloc(size_t alignment, size_t size) -> void*
    {
        return ::operator new(size, std::align_val_t{alignment});
    }

    ALPAKA_FN_INLINE ALPAKA_FN_HOST void alignedFree(size_t alignment, void* ptr)
    {
        ::operator delete(ptr, std::align_val_t{alignment});
    }
} // namespace alpaka::core
