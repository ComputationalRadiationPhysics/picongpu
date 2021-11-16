/* Copyright 2021 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Jeffrey Kelling
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
#    include <alpaka/math/isnan/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in isnan.
        class IsnanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsnan, IsnanUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA isnan trait specialization.
            template<typename TArg>
            struct Isnan<IsnanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(IsnanUniformCudaHipBuiltIn const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    return ::isnan(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
