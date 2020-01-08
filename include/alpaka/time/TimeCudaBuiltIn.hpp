/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

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
        class TimeCudaBuiltIn : public concepts::Implements<ConceptTime, TimeCudaBuiltIn>
        {
        public:
            //-----------------------------------------------------------------------------
            TimeCudaBuiltIn() = default;
            //-----------------------------------------------------------------------------
            __device__ TimeCudaBuiltIn(TimeCudaBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            __device__ TimeCudaBuiltIn(TimeCudaBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(TimeCudaBuiltIn const &) -> TimeCudaBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            __device__ auto operator=(TimeCudaBuiltIn &&) -> TimeCudaBuiltIn & = delete;
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
                __device__ static auto clock(
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
