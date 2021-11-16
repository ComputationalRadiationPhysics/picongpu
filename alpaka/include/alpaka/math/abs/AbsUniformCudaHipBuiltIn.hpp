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
#    include <alpaka/math/abs/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in abs.
        class AbsUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAbs, AbsUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA built in abs trait specialization.
            template<typename TArg>
            struct Abs<AbsUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(AbsUniformCudaHipBuiltIn const& abs_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(abs_ctx);
                    return ::abs(arg);
                }
            };
            //! The CUDA built in abs double specialization.
            template<>
            struct Abs<AbsUniformCudaHipBuiltIn, double>
            {
                __device__ auto operator()(AbsUniformCudaHipBuiltIn const& abs_ctx, double const& arg)
                {
                    alpaka::ignore_unused(abs_ctx);
                    return ::fabs(arg);
                }
            };
            //! The CUDA built in abs float specialization.
            template<>
            struct Abs<AbsUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(AbsUniformCudaHipBuiltIn const& abs_ctx, float const& arg) -> float
                {
                    alpaka::ignore_unused(abs_ctx);
                    return ::fabsf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
