/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Matthias Werner, Valentin Gehrke
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

#include <alpaka/math/rsqrt/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
    #include <cuda_runtime_api.h>
#else
    #if BOOST_COMP_HCC || BOOST_COMP_HIP
        #include <math_functions.h>
    #else
        #include <math_functions.hpp>
    #endif
#endif

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The HIP rsqrt.
        class RsqrtHipBuiltIn : public concepts::Implements<ConceptMathRsqrt, RsqrtHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The HIP rsqrt trait specialization.
            template<
                typename TArg>
            struct Rsqrt<
                RsqrtHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                __device__ static auto rsqrt(
                    RsqrtHipBuiltIn const & rsqrt_ctx,
                    TArg const & arg)
                -> decltype(::rsqrt(arg))
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return ::rsqrt(arg);
                }
            };
            //! The HIP rsqrt float specialization.
            template<>
            struct Rsqrt<
                RsqrtHipBuiltIn,
                float>
            {
                __device__ static auto rsqrt(
                    RsqrtHipBuiltIn const & rsqrt_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return ::rsqrtf(arg);
                }
            };
        }
    }
}

#endif
