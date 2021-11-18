/* Copyright 2019 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Bert Wesarg
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
#    include <alpaka/math/min/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in min.
        class MinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMin, MinUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA integral min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<
                MinUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<std::is_integral<Tx>::value && std::is_integral<Ty>::value>>
            {
                __device__ auto operator()(MinUniformCudaHipBuiltIn const& min_ctx, Tx const& x, Ty const& y)
                    -> decltype(::min(x, y))
                {
                    alpaka::ignore_unused(min_ctx);
                    return ::min(x, y);
                }
            };
            //! The standard library mixed integral floating point min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<
                MinUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_arithmetic<Tx>::value && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value && std::is_integral<Ty>::value)>>
            {
                __device__ auto operator()(MinUniformCudaHipBuiltIn const& min_ctx, Tx const& x, Ty const& y)
                    -> decltype(::fmin(x, y))
                {
                    alpaka::ignore_unused(min_ctx);
                    return ::fmin(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
