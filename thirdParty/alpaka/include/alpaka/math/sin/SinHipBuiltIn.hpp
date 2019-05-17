/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Matthias Werner, Valentin Gehrke
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

#include <alpaka/math/sin/Traits.hpp>   // Sin



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
        //! The standard library sin.
        class SinHipBuiltIn
        {
        public:
            using SinBase = SinHipBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library sin trait specialization.
            template<
                typename TArg>
            struct Sin<
                SinHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto sin(
                    SinHipBuiltIn const & sin_ctx,
                    TArg const & arg)
                -> decltype(::sin(arg))
                {
                    alpaka::ignore_unused(sin_ctx);
                    return ::sin(arg);
                }
            };
        }
    }
}

#endif
