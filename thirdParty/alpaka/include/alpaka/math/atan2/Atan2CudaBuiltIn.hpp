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

#include <alpaka/math/atan2/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in atan2.
        class Atan2CudaBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2CudaBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA atan2 trait specialization.
            template<
                typename Ty,
                typename Tx>
            struct Atan2<
                Atan2CudaBuiltIn,
                Ty,
                Tx,
                typename std::enable_if<
                    std::is_floating_point<Ty>::value
                    && std::is_floating_point<Tx>::value>::type>
            {
                __device__ static auto atan2(
                    Atan2CudaBuiltIn const & atan2_ctx,
                    Ty const & y,
                    Tx const & x)
                -> decltype(::atan2(y, x))
                {
                    alpaka::ignore_unused(atan2_ctx);
                    return ::atan2(y, x);
                }
            };

            template<>
            struct Atan2<
                Atan2CudaBuiltIn,
                float,
                float>
            {
                __device__ static auto atan2(
                    Atan2CudaBuiltIn const & atan2_ctx,
                    float const & y,
                    float const & x)
                -> float
                {
                    alpaka::ignore_unused(atan2_ctx);
                    return ::atan2f(y, x);
                }
            };
        }
    }
}

#endif
