/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/math/remainder/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
    #include <cuda_runtime_api.h>
#else
    #if BOOST_COMP_HCC || BOOST_COMP_HIP
        #include <math_functions.h>
    #else
        #include <math_functions.hpp>
    #endif
#endif

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The HIP remainder.
        class RemainderHipBuiltIn : public concepts::Implements<ConceptMathRemainder, RemainderHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The HIP remainder trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Remainder<
                RemainderHipBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_floating_point<Tx>::value
                    && std::is_floating_point<Ty>::value>::type>
            {
                __device__ static auto remainder(
                    RemainderHipBuiltIn const & remainder_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::remainder(x, y))
                {
                    alpaka::ignore_unused(remainder_ctx);
                    return ::remainder(x, y);
                }
            };
            //! The HIP remainder float specialization.
            template<>
            struct Remainder<
                RemainderHipBuiltIn,
                float,
                float>
            {
                __device__ static auto remainder(
                    RemainderHipBuiltIn const & remainder_ctx,
                    float const & x,
                    float const & y)
                -> float
                {
                    alpaka::ignore_unused(remainder_ctx);
                    return ::remainderf(
                        x,
                        y);
                }
            };
        }
    }
}

#endif
