/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/core/Positioning.hpp>
#include <alpaka/atomic/Traits.hpp>

#include <type_traits>

namespace alpaka
{
    namespace atomic
    {
        namespace atomicHierarchy
        {
            class Empty0{};
            class Empty1{};
            class Empty2{};
        }
        //#############################################################################
        //! build a single class to inherit from different atomic implementations
        //
        //  This implementation inherit from all three hierarchies.
        //  The multiple usage of the same type for different levels is allowed.
        //  The class provide the feature that each atomic operation can be focused
        //  to a hierarchy level in Alpaka. A operation to a hierarchy is independent
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
        class AtomicHierarchy :
            public TGridAtomic,
            public std::conditional<
                std::is_same<TGridAtomic,TBlockAtomic>::value,
                atomicHierarchy::Empty1,
                TBlockAtomic
            >::type,
            public std::conditional<
                std::is_same<TGridAtomic,TThreadAtomic>::value ||
                    std::is_same<TBlockAtomic,TThreadAtomic>::value,
                atomicHierarchy::Empty2,
                TThreadAtomic
            >::type
        {
            public:
            using UsedAtomicHierarchies = AtomicHierarchy<
                TGridAtomic,
                TBlockAtomic,
                TThreadAtomic
            >;
        };

        namespace traits
        {
            template<
                typename TGridAtomic,
                typename TBlockAtomic,
                typename TThreadAtomic
            >
            struct AtomicBase<
                AtomicHierarchy<
                    TGridAtomic,
                    TBlockAtomic,
                    TThreadAtomic
                >,
                hierarchy::Threads>
            {
                using type = TThreadAtomic;
            };

            template<
                typename TGridAtomic,
                typename TBlockAtomic,
                typename TThreadAtomic
            >
            struct AtomicBase<
                AtomicHierarchy<
                    TGridAtomic,
                    TBlockAtomic,
                    TThreadAtomic
                >,
                hierarchy::Blocks>
            {
                using type = TBlockAtomic;
            };

            template<
                typename TGridAtomic,
                typename TBlockAtomic,
                typename TThreadAtomic
            >
            struct AtomicBase<
                AtomicHierarchy<
                    TGridAtomic,
                    TBlockAtomic,
                    TThreadAtomic
                >,
                hierarchy::Grids>
            {
                using type = TGridAtomic;
            };

        }
    }
}
