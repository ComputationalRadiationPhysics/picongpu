/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Matthias Werner
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

#include <alpaka/math/pow/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
    #include <cuda_runtime_api.h>
#else
    #if BOOST_COMP_HCC
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
        //! The HIP pow.
        class PowHipBuiltIn
        {
        public:
            using PowBase = PowHipBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The HIP pow trait specialization.
            template<
                typename TBase,
                typename TExp>
            struct Pow<
                PowHipBuiltIn,
                TBase,
                TExp,
                typename std::enable_if<
                    std::is_floating_point<TBase>::value
                    && std::is_floating_point<TExp>::value>::type>
            {
                __device__ static auto pow(
                    PowHipBuiltIn const & pow_ctx,
                    TBase const & base,
                    TExp const & exp)
                -> decltype(::pow(base, exp))
                {
                    alpaka::ignore_unused(pow_ctx);
                    return ::pow(base, exp);
                }
            };
            //! The HIP pow float specialization.
            template<>
            struct Pow<
                PowHipBuiltIn,
                float,
                float>
            {
                __device__ static auto pow(
                    PowHipBuiltIn const & pow_ctx,
                    float const & base,
                    float const & exp)
                -> float
                {
                    alpaka::ignore_unused(pow_ctx);
                    return ::powf(base, exp);
                }
            };
        }
    }
}

#endif
