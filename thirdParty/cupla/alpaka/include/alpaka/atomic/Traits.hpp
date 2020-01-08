/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/meta/IsStrictBase.hpp>

#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Common.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The atomic operation traits specifics.
    namespace atomic
    {
        //-----------------------------------------------------------------------------
        //! The atomic operation traits.
        namespace traits
        {
            //#############################################################################
            //! The atomic operation trait.
            template<
                typename TOp,
                typename TAtomic,
                typename T,
                typename THierarchy,
                typename TSfinae = void>
            struct AtomicOp;

            //#############################################################################
            //! Get the atomic implementation for a hierarchy level
            template<
                typename TAtomic,
                typename THierarchy
            >
            struct AtomicBase;

        }

        //-----------------------------------------------------------------------------
        //! Executes the given operation atomically.
        //!
        //! \tparam TOp The operation type.
        //! \tparam T The value type.
        //! \tparam TAtomic The atomic implementation type.
        //! \param addr The value to change atomically.
        //! \param value The value used in the atomic operation.
        //! \param atomic The atomic implementation.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOp,
            typename TAtomic,
            typename T,
            typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicOp(
            TAtomic const & atomic,
            T * const addr,
            T const & value,
            THierarchy const & = THierarchy())
        -> T
        {
            return
                traits::AtomicOp<
                    TOp,
                    TAtomic,
                    T,
                    THierarchy>
                ::atomicOp(
                    atomic,
                    addr,
                    value);
        }

        //-----------------------------------------------------------------------------
        //! Executes the given operation atomically.
        //!
        //! \tparam TOp The operation type.
        //! \tparam TAtomic The atomic implementation type.
        //! \tparam T The value type.
        //! \param atomic The atomic implementation.
        //! \param addr The value to change atomically.
        //! \param compare The comparison value used in the atomic operation.
        //! \param value The value used in the atomic operation.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TOp,
            typename TAtomic,
            typename T,
            typename THierarchy = hierarchy::Grids>
        ALPAKA_FN_HOST_ACC auto atomicOp(
            TAtomic const & atomic,
            T * const addr,
            T const & compare,
            T const & value,
            THierarchy const & = THierarchy())
        -> T
        {
            return
                traits::AtomicOp<
                    TOp,
                    TAtomic,
                    T,
                    THierarchy>
                ::atomicOp(
                    atomic,
                    addr,
                    compare,
                    value);
        }

        namespace traits
        {
            //#############################################################################
            //! The AtomicOp trait specialization for classes with `UsedAtomicHierarchies` member type.
            template<
                typename TOp,
                typename TAtomic,
                typename T,
                typename THierarchy>
            struct AtomicOp<
                TOp,
                TAtomic,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto atomicOp(
                    TAtomic const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // Delegate the call to the base class.
                    return
                        atomic::atomicOp<
                            TOp>(
                                static_cast<
                                    typename AtomicBase<
                                        typename TAtomic::UsedAtomicHierarchies,
                                        THierarchy
                                    >::type const &>(atomic),
                                addr,
                                value,
                                THierarchy());
                }
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto atomicOp(
                    TAtomic const & atomic,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    // Delegate the call to the base class.
                    return
                        atomic::atomicOp<
                            TOp>(
                                static_cast<
                                    typename AtomicBase<
                                        typename TAtomic::UsedAtomicHierarchies,
                                        THierarchy
                                    >::type const &>(atomic),
                                addr,
                                compare,
                                value,
                                THierarchy());
                }
            };
        }
    }
}
