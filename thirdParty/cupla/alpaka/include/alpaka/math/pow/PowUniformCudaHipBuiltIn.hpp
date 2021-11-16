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
#    include <alpaka/math/pow/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in pow.
        class PowUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathPow, PowUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA pow trait specialization.
            template<typename TBase, typename TExp>
            struct Pow<
                PowUniformCudaHipBuiltIn,
                TBase,
                TExp,
                std::enable_if_t<std::is_floating_point<TBase>::value && std::is_floating_point<TExp>::value>>
            {
                __device__ auto operator()(PowUniformCudaHipBuiltIn const& pow_ctx, TBase const& base, TExp const& exp)
                {
                    alpaka::ignore_unused(pow_ctx);
                    return ::pow(base, exp);
                }
            };
            //! The CUDA pow float specialization.
            template<>
            struct Pow<PowUniformCudaHipBuiltIn, float, float>
            {
                __device__ auto operator()(
                    PowUniformCudaHipBuiltIn const& pow_ctx,
                    float const& base,
                    float const& exp) -> float
                {
                    alpaka::ignore_unused(pow_ctx);
                    return ::powf(base, exp);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
