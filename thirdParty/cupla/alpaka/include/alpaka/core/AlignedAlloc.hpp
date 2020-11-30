/* Copyright 2020 René Widera
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

#if BOOST_COMP_MSVC
#    include <malloc.h>
#else
#    include <cstdlib>
#endif

namespace alpaka
{
    namespace core
    {
        //-----------------------------------------------------------------------------
        //! Rounds to the next higher power of two (if not already power of two).
        // Adapted from llvm/ADT/SmallPtrSet.h
        ALPAKA_FN_INLINE ALPAKA_FN_HOST void* alignedAlloc(size_t alignment, size_t size)
        {
#if BOOST_OS_WINDOWS
            return _aligned_malloc(size, alignment);
#elif BOOST_OS_MACOS
            void* ptr = nullptr;
            posix_memalign(&ptr, alignment, size);
            return ptr;
#else
            // the amount of bytes to allocate must be a multiple of the alignment
            size_t sizeToAllocate = ((size + alignment - 1u) / alignment) * alignment;
            return ::aligned_alloc(alignment, sizeToAllocate);
#endif
        }

        ALPAKA_FN_INLINE ALPAKA_FN_HOST void alignedFree(void* ptr)
        {
#if BOOST_OS_WINDOWS
            _aligned_free(ptr);
#else
            // linux and macos
            ::free(ptr);
#endif
        }

        //#############################################################################
        //! destroy aligned object and free aligned memory
        struct AlignedDelete
        {
            constexpr AlignedDelete() = default;

            //-----------------------------------------------------------------------------
            //! Calls ~T() on ptr to destroy the object and then calls aligned_free to free the allocated memory.
            template<typename T>
            void operator()(T* ptr) const
            {
                if(ptr)
                    ptr->~T();
                alignedFree(reinterpret_cast<void*>(ptr));
            }
        };
    } // namespace core
} // namespace alpaka
