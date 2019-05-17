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

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/round/Traits.hpp>

#include <cuda_runtime.h>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library round.
        class RoundCudaBuiltIn
        {
        public:
            using RoundBase = RoundCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Round<
                RoundCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto round(
                    RoundCudaBuiltIn const & round_ctx,
                    TArg const & arg)
                -> decltype(::round(arg))
                {
                    alpaka::ignore_unused(round_ctx);
                    return ::round(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Lround<
                RoundCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto lround(
                    RoundCudaBuiltIn const & lround_ctx,
                    TArg const & arg)
                -> long int
                {
                    alpaka::ignore_unused(lround_ctx);
                    return ::lround(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Llround<
                RoundCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto llround(
                    RoundCudaBuiltIn const & llround_ctx,
                    TArg const & arg)
                -> long int
                {
                    alpaka::ignore_unused(llround_ctx);
                    return ::llround(arg);
                }
            };
        }
    }
}

#endif
