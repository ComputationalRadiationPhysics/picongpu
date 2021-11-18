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
#    include <alpaka/math/sin/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA sin.
        class SinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSin, SinUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA sin trait specialization.
            template<typename TArg>
            struct Sin<SinUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(SinUniformCudaHipBuiltIn const& sin_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sin(arg);
                }
            };
            //! The CUDA sin float specialization.
            template<>
            struct Sin<SinUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(SinUniformCudaHipBuiltIn const& sin_ctx, float const& arg) -> float
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sinf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
