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

#include <alpaka/math/fmod/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
    #include <cuda_runtime_api.h>
#else
    #if BOOST_COMP_HCC
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
        //! The HIP fmod.
        class FmodHipBuiltIn
        {
        public:
            using FmodBase = FmodHipBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The HIP fmod trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Fmod<
                FmodHipBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_floating_point<Tx>::value
                    && std::is_floating_point<Ty>::value>::type>
            {
                __device__ static auto fmod(
                    FmodHipBuiltIn const & fmod_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::fmod(x, y))
                {
                    alpaka::ignore_unused(fmod_ctx);
                    return ::fmod(x, y);
                }
            };
            //! The HIP fmod float specialization.
            template<>
            struct Fmod<
                FmodHipBuiltIn,
                float,
                float>
            {
                __device__ static auto fmod(
                    FmodHipBuiltIn const & fmod_ctx,
                    float const & x,
                    float const & y)
                -> float
                {
                    alpaka::ignore_unused(fmod_ctx);
                    return ::fmodf(x, y);
                }
            };
        }
    }
}

#endif
