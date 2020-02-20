/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/fmod/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in fmod.
        class FmodCudaBuiltIn : public concepts::Implements<ConceptMathFmod, FmodCudaBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA fmod trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Fmod<
                FmodCudaBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_floating_point<Tx>::value
                    && std::is_floating_point<Ty>::value>::type>
            {
                __device__ static auto fmod(
                    FmodCudaBuiltIn const & fmod_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::fmod(x, y))
                {
                    alpaka::ignore_unused(fmod_ctx);
                    return ::fmod(
                        x,
                        y);
                }
            };
            //! The CUDA fmod float specialization.
            template<>
            struct Fmod<
                FmodCudaBuiltIn,
                float,
                float>
            {
                __device__ static auto fmod(
                    FmodCudaBuiltIn const & fmod_ctx,
                    float const & x,
                    float const & y)
                -> float
                {
                    alpaka::ignore_unused(fmod_ctx);
                    return ::fmodf(
                        x,
                        y);
                }
            };
        }
    }
}

#endif
