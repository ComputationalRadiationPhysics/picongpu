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
#    include <alpaka/math/atan/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in atan.
        class AtanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan, AtanUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA atan trait specialization.
            template<typename TArg>
            struct Atan<AtanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(AtanUniformCudaHipBuiltIn const& atan_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(atan_ctx);
                    return ::atan(arg);
                }
            };

            template<>
            struct Atan<AtanUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(AtanUniformCudaHipBuiltIn const& atan_ctx, float const& arg) -> float
                {
                    alpaka::ignore_unused(atan_ctx);
                    return ::atanf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
