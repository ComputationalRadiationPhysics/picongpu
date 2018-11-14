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

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*, __HIPCC__

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/time/Traits.hpp>       // time::Clock

namespace alpaka
{
    namespace time
    {
        //#############################################################################
        //! The GPU HIP accelerator time implementation.
        class TimeHipBuiltIn
        {
        public:
            using TimeBase = TimeHipBuiltIn;

            //-----------------------------------------------------------------------------
            //! Default constructor.
            ALPAKA_FN_HOST_ACC TimeHipBuiltIn() = default;
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            __device__ TimeHipBuiltIn(TimeHipBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            __device__ TimeHipBuiltIn(TimeHipBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            __device__ auto operator=(TimeHipBuiltIn const &) -> TimeHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            __device__ auto operator=(TimeHipBuiltIn &&) -> TimeHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            //! Destructor.
            /*virtual*/ ALPAKA_FN_HOST_ACC ~TimeHipBuiltIn() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The HIP built-in clock operation.
            template<>
            struct Clock<
                time::TimeHipBuiltIn>
            {
                //-----------------------------------------------------------------------------

                __device__ static auto clock(
                    time::TimeHipBuiltIn const &)
                -> std::uint64_t
                {
                    // This can be converted to a wall-clock time in seconds by dividing through the shader clock rate given by hipDeviceProp::clockRate.
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
