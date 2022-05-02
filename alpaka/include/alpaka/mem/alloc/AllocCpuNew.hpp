/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/mem/alloc/Traits.hpp>

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
