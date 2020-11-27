/* Copyright 2019 René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENMP

#    include <alpaka/atomic/Op.hpp>
#    include <alpaka/atomic/Traits.hpp>

namespace alpaka
{
    //#############################################################################
    //! The OpenMP accelerators atomic ops.
    //
    //  Atomics can be used in the blocks and threads hierarchy levels.
    //  Atomics are not guaranteed to be safe between devices or grids.
    class AtomicOmpBuiltIn
    {
    public:
        //-----------------------------------------------------------------------------
        AtomicOmpBuiltIn() = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST AtomicOmpBuiltIn(AtomicOmpBuiltIn const&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST AtomicOmpBuiltIn(AtomicOmpBuiltIn&&) = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(AtomicOmpBuiltIn const&) -> AtomicOmpBuiltIn& = delete;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator=(AtomicOmpBuiltIn&&) -> AtomicOmpBuiltIn& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~AtomicOmpBuiltIn() = default;
    };

    namespace traits
    {
// check for OpenMP 3.1+
// "omp atomic capture" is not supported before OpenMP 3.1
#    if _OPENMP >= 201107

        //#############################################################################
        //! The OpenMP accelerators atomic operation: ADD
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicAdd, AtomicOmpBuiltIn, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        pragma omp atomic capture
                {
                    old = ref;
                    ref += value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenMP accelerators atomic operation: SUB
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicSub, AtomicOmpBuiltIn, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        pragma omp atomic capture
                {
                    old = ref;
                    ref -= value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenMP accelerators atomic operation: EXCH
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicExch, AtomicOmpBuiltIn, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        pragma omp atomic capture
                {
                    old = ref;
                    ref = value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenMP accelerators atomic operation: AND
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicAnd, AtomicOmpBuiltIn, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        pragma omp atomic capture
                {
                    old = ref;
                    ref &= value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenMP accelerators atomic operation: OR
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicOr, AtomicOmpBuiltIn, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        pragma omp atomic capture
                {
                    old = ref;
                    ref |= value;
                }
                return old;
            }
        };

        //#############################################################################
        //! The OpenMP accelerators atomic operation: XOR
        template<typename T, typename THierarchy>
        struct AtomicOp<AtomicXor, AtomicOmpBuiltIn, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
            {
                T old;
                auto& ref(*addr);
// atomically update ref, but capture the original value in old
#        pragma omp atomic capture
                {
                    old = ref;
                    ref ^= value;
                }
                return old;
            }
        };

#    endif // _OPENMP >= 201107

        //#############################################################################
        //! The OpenMP accelerators atomic operation
        //
        // generic implementations for operations where native atomics are not available
        template<typename TOp, typename T, typename THierarchy>
        struct AtomicOp<TOp, AtomicOmpBuiltIn, T, THierarchy>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(AtomicOmpBuiltIn const&, T* const addr, T const& value) -> T
            {
                T old;
// \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all
// threads.
#    pragma omp critical(AlpakaOmpAtomicOp)
                {
                    old = TOp()(addr, value);
                }
                return old;
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto atomicOp(
                AtomicOmpBuiltIn const&,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                T old;
// \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all
// threads.
#    pragma omp critical(AlpakaOmpAtomicOp2)
                {
                    old = TOp()(addr, compare, value);
                }
                return old;
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
