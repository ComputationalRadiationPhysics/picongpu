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

#include <alpaka/math/sin/Traits.hpp>

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
        //! The HIP sin.
        class SinHipBuiltIn : public concepts::Implements<ConceptMathSin, SinHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The HIP sin trait specialization.
            template<
                typename TArg>
            struct Sin<
                SinHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto sin(
                    SinHipBuiltIn const & sin_ctx,
                    TArg const & arg)
                -> decltype(::sin(arg))
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sin(arg);
                }
            };
            //! The HIP sin float specialization.
            template<>
            struct Sin<
                SinHipBuiltIn,
                float>
            {
                __device__ static auto sin(
                    SinHipBuiltIn const & sin_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sinf(arg);
                }
            };
        }
    }
}

#endif
