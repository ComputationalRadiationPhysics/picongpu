/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/time/Traits.hpp>

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
