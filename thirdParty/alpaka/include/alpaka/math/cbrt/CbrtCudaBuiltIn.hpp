/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/cbrt/Traits.hpp>

#include <type_traits>
#include <cmath>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library cbrt.
        class CbrtCudaBuiltIn
        {
        public:
            using CbrtBase = CbrtCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library cbrt trait specialization.
            template<
                typename TArg>
            struct Cbrt<
                CbrtCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_arithmetic<TArg>::value>::type>
            {
                __device__ static auto cbrt(
                    CbrtCudaBuiltIn const & cbrt_ctx,
                    TArg const & arg)
                -> decltype(::cbrt(arg))
                {
                    alpaka::ignore_unused(cbrt_ctx);
                    return ::cbrt(arg);
                }
            };
        }
    }
}

#endif
