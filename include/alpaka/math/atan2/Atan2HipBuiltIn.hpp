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

#include <alpaka/math/atan2/Traits.hpp>

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
        //! The HIP atan2.
        class Atan2HipBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2HipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The HIP atan2 trait specialization.
            template<
                typename Ty,
                typename Tx>
            struct Atan2<
                Atan2HipBuiltIn,
                Ty,
                Tx,
                typename std::enable_if<
                    std::is_floating_point<Ty>::value
                    && std::is_floating_point<Tx>::value>::type>
            {
                __device__ static auto atan2(
                    Atan2HipBuiltIn const & atan2_ctx,
                    Ty const & y,
                    Tx const & x)
                -> decltype(::atan2(y, x))
                {
                    alpaka::ignore_unused(atan2_ctx);
                    return ::atan2(y, x);
                }
            };
            //! The HIP sin float specialization.
            template<>
            struct Atan2<
                Atan2HipBuiltIn,
                float,
                float>
            {
                __device__ static auto atan2(
                    Atan2HipBuiltIn const & atan2_ctx,
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
