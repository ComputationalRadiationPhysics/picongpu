/* Copyright 2019 Benjamin Worpitz, Matthias Werner
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

#include <alpaka/math/sincos/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA sincos.
        class SinCosCudaBuiltIn : public concepts::Implements<ConceptMathSinCos, SinCosCudaBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################

            //! sincos trait specialization.
            template<>
            struct SinCos<
                SinCosCudaBuiltIn,
                double>
            {
                __device__ static auto sincos(
                    SinCosCudaBuiltIn const & sincos_ctx,
                    double const & arg,
                    double & result_sin,
                    double & result_cos)
                -> void
                {
                    alpaka::ignore_unused(sincos_ctx);
                    ::sincos(arg, &result_sin, &result_cos);
                }
            };

            //! The CUDA sin float specialization.
            template<>
            struct SinCos<
                SinCosCudaBuiltIn,
                float>
            {
                __device__ static auto sincos(
                    SinCosCudaBuiltIn const & sincos_ctx,
                    float const & arg,
                    float & result_sin,
                    float & result_cos)
                -> void
                {
                    alpaka::ignore_unused(sincos_ctx);
                    ::sincosf(arg, &result_sin, &result_cos);
                }
            };

        }
    }
}

#endif
