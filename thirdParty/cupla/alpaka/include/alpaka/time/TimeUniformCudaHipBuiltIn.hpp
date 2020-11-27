/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/time/Traits.hpp>

namespace alpaka
{
    //#############################################################################
    //! The GPU CUDA accelerator time implementation.
    class TimeUniformCudaHipBuiltIn : public concepts::Implements<ConceptTime, TimeUniformCudaHipBuiltIn>
    {
    public:
        //-----------------------------------------------------------------------------
        TimeUniformCudaHipBuiltIn() = default;
        //-----------------------------------------------------------------------------
        __device__ TimeUniformCudaHipBuiltIn(TimeUniformCudaHipBuiltIn const&) = delete;
        //-----------------------------------------------------------------------------
        __device__ TimeUniformCudaHipBuiltIn(TimeUniformCudaHipBuiltIn&&) = delete;
        //-----------------------------------------------------------------------------
        __device__ auto operator=(TimeUniformCudaHipBuiltIn const&) -> TimeUniformCudaHipBuiltIn& = delete;
        //-----------------------------------------------------------------------------
        __device__ auto operator=(TimeUniformCudaHipBuiltIn&&) -> TimeUniformCudaHipBuiltIn& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~TimeUniformCudaHipBuiltIn() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The CUDA built-in clock operation.
        template<>
        struct Clock<TimeUniformCudaHipBuiltIn>
        {
            //-----------------------------------------------------------------------------
            __device__ static auto clock(TimeUniformCudaHipBuiltIn const&) -> std::uint64_t
            {
                // This can be converted to a wall-clock time in seconds by dividing through the shader clock rate
                // given by uniformCudaHipDeviceProp::clockRate. This clock rate is double the main clock rate on Fermi
                // and older cards.
                return static_cast<std::uint64_t>(clock64());
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
