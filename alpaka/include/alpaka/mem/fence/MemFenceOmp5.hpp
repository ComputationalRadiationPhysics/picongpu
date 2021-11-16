/* Copyright 2021 Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/mem/fence/Traits.hpp>

namespace alpaka
{
    //! The OpenMP 5 block memory fence.
    class MemFenceOmp5 : public concepts::Implements<ConceptMemFence, MemFenceOmp5>
    {
    };

    namespace traits
    {
        template<typename TMemScope>
        struct MemFence<MemFenceOmp5, TMemScope>
        {
            static auto mem_fence(MemFenceOmp5 const&, TMemScope const&)
            {
                // We only have one fence scope available in OpenMP 5 which encompasses the whole device
#    pragma omp flush acq_rel
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
