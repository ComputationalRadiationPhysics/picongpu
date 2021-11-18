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
#    include <alpaka/math/fmod/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The CUDA built in fmod.
        class FmodUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFmod, FmodUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //! The CUDA fmod trait specialization.
            template<typename Tx, typename Ty>
            struct Fmod<
                FmodUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<std::is_floating_point<Tx>::value && std::is_floating_point<Ty>::value>>
            {
                __device__ auto operator()(FmodUniformCudaHipBuiltIn const& fmod_ctx, Tx const& x, Ty const& y)
                {
                    alpaka::ignore_unused(fmod_ctx);
                    return ::fmod(x, y);
                }
            };
            //! The CUDA fmod float specialization.
            template<>
            struct Fmod<FmodUniformCudaHipBuiltIn, float, float>
            {
                __device__ auto operator()(FmodUniformCudaHipBuiltIn const& fmod_ctx, float const& x, float const& y)
                    -> float
                {
                    alpaka::ignore_unused(fmod_ctx);
                    return ::fmodf(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
