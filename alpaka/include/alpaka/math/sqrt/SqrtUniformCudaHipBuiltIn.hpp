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
#    include <alpaka/math/sqrt/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA sqrt.
        class SqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSqrt, SqrtUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA sqrt trait specialization.
            template<typename TArg>
            struct Sqrt<SqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(SqrtUniformCudaHipBuiltIn const& sqrt_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(sqrt_ctx);
                    return ::sqrt(arg);
                }
            };
            //! The CUDA sqrt float specialization.
            template<>
            struct Sqrt<SqrtUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(SqrtUniformCudaHipBuiltIn const& sqrt_ctx, float const& arg) -> float
                {
                    alpaka::ignore_unused(sqrt_ctx);
                    return ::sqrtf(arg);
                }
            };

        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
