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

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/math/max/Traits.hpp>   // Max



#include <type_traits>
#if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
    #include <cuda_runtime_api.h>
#else
    #if BOOST_COMP_HCC
        #include <math_functions.h>
    #else
        #include <math_functions.hpp>
    #endif
#endif

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library max.
        class MaxHipBuiltIn
        {
        public:
            using MaxBase = MaxHipBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library integral max trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Max<
                MaxHipBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_integral<Tx>::value
                    && std::is_integral<Ty>::value>::type>
            {
                __device__ static auto max(
                    MaxHipBuiltIn const & max_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::max(x, y))
                {
                    alpaka::ignore_unused(max_ctx);
                    return ::max(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point max trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Max<
                MaxHipBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_arithmetic<Tx>::value
                    && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value
                        && std::is_integral<Ty>::value)>::type>
            {
                __device__ static auto max(
                    MaxHipBuiltIn const & max_ctx,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::fmax(x, y))
                {
                    alpaka::ignore_unused(max_ctx);
                    return ::fmax(x, y);
                }
            };
        }
    }
}

#endif
