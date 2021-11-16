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
#    include <alpaka/math/atan2/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in atan2.
        class Atan2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2UniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA atan2 trait specialization.
            template<typename Ty, typename Tx>
            struct Atan2<
                Atan2UniformCudaHipBuiltIn,
                Ty,
                Tx,
                std::enable_if_t<std::is_floating_point<Ty>::value && std::is_floating_point<Tx>::value>>
            {
                __device__ auto operator()(Atan2UniformCudaHipBuiltIn const& atan2_ctx, Ty const& y, Tx const& x)
                {
                    alpaka::ignore_unused(atan2_ctx);
                    return ::atan2(y, x);
                }
            };

            template<>
            struct Atan2<Atan2UniformCudaHipBuiltIn, float, float>
            {
                __device__ auto operator()(Atan2UniformCudaHipBuiltIn const& atan2_ctx, float const& y, float const& x)
                    -> float
                {
                    alpaka::ignore_unused(atan2_ctx);
                    return ::atan2f(y, x);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
