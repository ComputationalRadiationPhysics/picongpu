/**
* \file
* Copyright 2014-2016 Benjamin Worpitz, Rene Widera
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

#include <alpaka/atomic/Traits.hpp>

#include <boost/core/ignore_unused.hpp>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The CPU fibers accelerator atomic ops.
        class AtomicNoOp
        {
        public:

            //-----------------------------------------------------------------------------
            AtomicNoOp() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicNoOp(AtomicNoOp const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA AtomicNoOp(AtomicNoOp &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicNoOp const &) -> AtomicNoOp & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_NO_CUDA auto operator=(AtomicNoOp &&) -> AtomicNoOp & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicNoOp() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The CPU fibers accelerator atomic operation.
            template<
                typename TOp,
                typename T,
                typename THierarchy>
            struct AtomicOp<
                TOp,
                atomic::AtomicNoOp,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicNoOp const & atomic,
                    T * const addr,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    return TOp()(addr, value);
                }
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA static auto atomicOp(
                    atomic::AtomicNoOp const & atomic,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    boost::ignore_unused(atomic);
                    return TOp()(addr, compare, value);
                }
            };
        }
    }
}
