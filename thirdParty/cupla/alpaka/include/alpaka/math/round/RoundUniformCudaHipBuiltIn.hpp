/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/CudaHipMath.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/math/round/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA round.
        class RoundUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRound, RoundUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA round trait specialization.
            template<typename TArg>
            struct Round<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(RoundUniformCudaHipBuiltIn const& round_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(round_ctx);
                    return ::round(arg);
                }
            };
            //! The CUDA lround trait specialization.
            template<typename TArg>
            struct Lround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(RoundUniformCudaHipBuiltIn const& lround_ctx, TArg const& arg) -> long int
                {
                    alpaka::ignore_unused(lround_ctx);
                    return ::lround(arg);
                }
            };
            //! The CUDA llround trait specialization.
            template<typename TArg>
            struct Llround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(RoundUniformCudaHipBuiltIn const& llround_ctx, TArg const& arg) -> long int
                {
                    alpaka::ignore_unused(llround_ctx);
                    return ::llround(arg);
                }
            };
            //! The CUDA round float specialization.
            template<>
            struct Round<RoundUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(RoundUniformCudaHipBuiltIn const& round_ctx, float const& arg) -> float
                {
                    alpaka::ignore_unused(round_ctx);
                    return ::roundf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
