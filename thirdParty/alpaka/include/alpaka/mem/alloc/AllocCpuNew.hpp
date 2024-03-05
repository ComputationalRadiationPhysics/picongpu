/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/mem/alloc/Traits.hpp"

namespace alpaka
{
    //! The CPU new allocator.
    class AllocCpuNew : public concepts::Implements<ConceptMemAlloc, AllocCpuNew>
    {
    };

    namespace trait
    {
        //! The CPU new allocator memory allocation trait specialization.
        template<typename T>
        struct Malloc<T, AllocCpuNew>
        {
            ALPAKA_FN_HOST static auto malloc(AllocCpuNew const& /* alloc */, std::size_t const& sizeElems) -> T*
            {
                return new T[sizeElems];
            }
        };

        //! The CPU new allocator memory free trait specialization.
        template<typename T>
        struct Free<T, AllocCpuNew>
        {
            ALPAKA_FN_HOST static auto free(AllocCpuNew const& /* alloc */, T const* const ptr) -> void
            {
                return delete[] ptr;
            }
        };
    } // namespace trait
} // namespace alpaka
