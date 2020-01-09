/* Copyright 2019 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Bert Wesarg
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

#include <alpaka/math/min/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in min.
        class MinCudaBuiltIn : public concepts::Implements<ConceptMathMin, MinCudaBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA integral min trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Min<
                MinCudaBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_integral<Tx>::value
                    && std::is_integral<Ty>::value>::type>
            {
                __device__ static auto min(
                    MinCudaBuiltIn const & min_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::min(x, y))
                {
                    alpaka::ignore_unused(min_ctx);
                    return ::min(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point min trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Min<
                MinCudaBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_arithmetic<Tx>::value
                    && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value
                        && std::is_integral<Ty>::value)>::type>
            {
                __device__ static auto min(
                    MinCudaBuiltIn const & min_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::fmin(x, y))
                {
                    alpaka::ignore_unused(min_ctx);
                    return ::fmin(x, y);
                }
            };
        }
    }
}

#endif
