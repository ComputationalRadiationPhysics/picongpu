/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/atomic/Traits.hpp>

#include <alpaka/meta/InheritFromList.hpp>
#include <alpaka/meta/Unique.hpp>

#include <tuple>

namespace alpaka
{
    namespace atomic
    {

        //#############################################################################
        //! build a single class to inherit from different atomic implementations
        //
        //  This implementation inherit from all three hierarchies.
        //  The multiple usage of the same type for different levels is allowed.
        //  The class provide the feature that each atomic operation can be focused
        //  to a hierarchy level in alpaka. A operation to a hierarchy is independent
        //  to the memory hierarchy.
        //
        //  \tparam TGridAtomic atomic implementation for atomic operations between grids within a device
        //  \tparam TBlockAtomic atomic implementation for atomic operations between blocks within a grid
        //  \tparam TThreadAtomic atomic implementation for atomic operations between threads within a block
        template<
            typename TGridAtomic,
            typename TBlockAtomic,
            typename TThreadAtomic
        >
        using AtomicHierarchy
            = alpaka::meta::InheritFromList<
                alpaka::meta::Unique<
                    std::tuple<
                        TGridAtomic,
                        TBlockAtomic,
                        TThreadAtomic,
                        concepts::Implements<ConceptAtomicGrids, TGridAtomic>,
                        concepts::Implements<ConceptAtomicBlocks, TBlockAtomic>,
                        concepts::Implements<ConceptAtomicThreads, TThreadAtomic>
                    >
                >
            >;
    }
}
