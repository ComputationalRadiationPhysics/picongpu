/**
* \file
* Copyright 2016 Rene Widera
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
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
