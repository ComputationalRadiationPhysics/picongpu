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
#    include <alpaka/math/max/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in max.
        class MaxUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMax, MaxUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The standard library integral max trait specialization.
            template<typename Tx, typename Ty>
            struct Max<
                MaxUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<std::is_integral<Tx>::value && std::is_integral<Ty>::value>>
            {
                __device__ auto operator()(MaxUniformCudaHipBuiltIn const& max_ctx, Tx const& x, Ty const& y)
                    -> decltype(::max(x, y))
                {
                    alpaka::ignore_unused(max_ctx);
                    return ::max(x, y);
                }
            };
            //! The CUDA mixed integral floating point max trait specialization.
            template<typename Tx, typename Ty>
            struct Max<
                MaxUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_arithmetic<Tx>::value && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value && std::is_integral<Ty>::value)>>
            {
                __device__ auto operator()(MaxUniformCudaHipBuiltIn const& max_ctx, Tx const& x, Ty const& y)
                    -> decltype(::fmax(x, y))
                {
                    alpaka::ignore_unused(max_ctx);
                    return ::fmax(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
