/**
 * \file
 * Copyright 2016 Benjamin Worpitz
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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/time/Traits.hpp>

namespace alpaka
{
    namespace time
    {
        //#############################################################################
        //! The GPU CUDA accelerator time implementation.
        class TimeCudaBuiltIn
        {
        public:
            using TimeBase = TimeCudaBuiltIn;

            //-----------------------------------------------------------------------------
            TimeCudaBuiltIn() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY TimeCudaBuiltIn(TimeCudaBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY TimeCudaBuiltIn(TimeCudaBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(TimeCudaBuiltIn const &) -> TimeCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_CUDA_ONLY auto operator=(TimeCudaBuiltIn &&) -> TimeCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~TimeCudaBuiltIn() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA built-in clock operation.
            template<>
            struct Clock<
                time::TimeCudaBuiltIn>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_CUDA_ONLY static auto clock(
                    time::TimeCudaBuiltIn const &)
                -> std::uint64_t
                {
                    // This can be converted to a wall-clock time in seconds by dividing through the shader clock rate given by cudaDeviceProp::clockRate.
                    // This clock rate is double the main clock rate on Fermi and older cards. 
                    return
                        static_cast<std::uint64_t>(
                            clock64());
                }
            };
        }
    }
}

#endif
