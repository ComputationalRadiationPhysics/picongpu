/**
* \file
* Copyright 2014-2018 Benjamin Worpitz, Rene Widera
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

#ifdef _OPENMP

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/atomic/Op.hpp>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The OpenMP accelerators atomic ops.
        //
        //  Atomics can be used in the blocks and threads hierarchy levels.
        //  Atomics are not guaranteed to be safe between devices or grids.
        class AtomicOmpBuiltIn
        {
        public:
            using AtomicBase = AtomicOmpBuiltIn;

            //-----------------------------------------------------------------------------
            AtomicOmpBuiltIn() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AtomicOmpBuiltIn(AtomicOmpBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AtomicOmpBuiltIn(AtomicOmpBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(AtomicOmpBuiltIn const &) -> AtomicOmpBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(AtomicOmpBuiltIn &&) -> AtomicOmpBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicOmpBuiltIn() = default;
        };

        namespace traits
        {

// check for OpenMP 3.1+
// "omp atomic capture" is not supported before OpenMP 3.1
#if _OPENMP >= 201107

            //#############################################################################
            //! The OpenMP accelerators atomic operation: ADD
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma omp atomic capture
                    {
                        old = ref;
                        ref += value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: SUB
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma omp atomic capture
                    {
                        old = ref;
                        ref -= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: EXCH
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma omp atomic capture
                    {
                        old = ref;
                        ref = value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: AND
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma omp atomic capture
                    {
                        old = ref;
                        ref &= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: OR
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma omp atomic capture
                    {
                        old = ref;
                        ref |= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: XOR
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma omp atomic capture
                    {
                        old = ref;
                        ref ^= value;
                    }
                    return old;
                }
            };

#endif // _OPENMP >= 201107

            //#############################################################################
            //! The OpenMP accelerators atomic operation
            //
            // generic implementations for operations where native atomics are not available
            template<
                typename TOp,
                typename T,
                typename THierarchy>
            struct AtomicOp<
                TOp,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    // \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all threads.
                    #pragma omp critical (AlpakaOmpAtomicOp)
                    {
                        old = TOp()(addr, value);
                    }
                    return old;
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    T old;
                    // \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all threads.
                    #pragma omp critical (AlpakaOmpAtomicOp2)
                    {
                        old = TOp()(addr, compare, value);
                    }
                    return old;
                }
            };
        }
    }
}

#endif
