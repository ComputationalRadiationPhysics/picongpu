/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <boost/core/ignore_unused.hpp>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The OpenMP accelerator atomic ops.
        //
        //  Atomics can be used in the blocks and threads hierarchy levels.
        //  Atomics are not guaranteed to be save between devices or grids.
        class AtomicOmpCritSec
        {
        public:
            using AtomicBase = AtomicOmpCritSec;

            //-----------------------------------------------------------------------------
            AtomicOmpCritSec() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicOmpCritSec(AtomicOmpCritSec const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicOmpCritSec(AtomicOmpCritSec &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicOmpCritSec const &) -> AtomicOmpCritSec & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicOmpCritSec &&) -> AtomicOmpCritSec & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicOmpCritSec() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The OpenMP accelerator atomic operation.
            //
            // NOTE: We can not use '#pragma omp atomic' because braces or calling other functions directly after '#pragma omp atomic' are not allowed.
            // So this would not be fully atomic. Between the store of the old value and the operation could be a context switch.
            template<
                typename TOp,
                typename T,
                typename THierarchy>
            struct AtomicOp<
                TOp,
                atomic::AtomicOmpCritSec,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicOmpCritSec const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    T old;
                    // \TODO: Currently not only the access to the same memory location is protected by a mutex but all atomic ops on all threads.
                    #pragma omp critical (AlpakaOmpAtomicOp)
                    {
                        old = TOp()(addr, value);
                    }
                    return old;
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicOmpCritSec const & atomic,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
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
