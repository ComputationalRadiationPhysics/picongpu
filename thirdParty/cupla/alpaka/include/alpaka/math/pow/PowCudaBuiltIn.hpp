/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
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

#include <alpaka/math/pow/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in pow.
        class PowCudaBuiltIn
        {
        public:
            using PowBase = PowCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA pow trait specialization.
            template<
                typename TBase,
                typename TExp>
            struct Pow<
                PowCudaBuiltIn,
                TBase,
                TExp,
                typename std::enable_if<
                    std::is_floating_point<TBase>::value
                    && std::is_floating_point<TExp>::value>::type>
            {
                __device__ static auto pow(
                    PowCudaBuiltIn const & pow_ctx,
                    TBase const & base,
                    TExp const & exp)
                -> decltype(::pow(base, exp))
                {
                    alpaka::ignore_unused(pow_ctx);
                    return ::pow(base, exp);
                }
            };
            //! The CUDA pow float specialization.
            template<>
            struct Pow<
                PowCudaBuiltIn,
                float,
                float>
            {
                __device__ static auto pow(
                    PowCudaBuiltIn const & pow_ctx,
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
