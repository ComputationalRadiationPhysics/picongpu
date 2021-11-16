/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Valentin Gehrke
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
#    include <alpaka/math/rsqrt/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA rsqrt.
        class RsqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRsqrt, RsqrtUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA rsqrt trait specialization.
            template<typename TArg>
            struct Rsqrt<RsqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                __device__ auto operator()(RsqrtUniformCudaHipBuiltIn const& rsqrt_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return ::rsqrt(arg);
                }
            };
            //! The CUDA rsqrt float specialization.
            template<>
            struct Rsqrt<RsqrtUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(RsqrtUniformCudaHipBuiltIn const& rsqrt_ctx, float const& arg) -> float
                {
                    alpaka::ignore_unused(rsqrt_ctx);
                    return ::rsqrtf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
