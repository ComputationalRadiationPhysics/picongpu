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

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/erf/Traits.hpp>

#include <cuda_runtime.h>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library erf.
        class ErfCudaBuiltIn
        {
        public:
            using ErfBase = ErfCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library erf trait specialization.
            template<
                typename TArg>
            struct Erf<
                ErfCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto erf(
                    ErfCudaBuiltIn const & erf_ctx,
                    TArg const & arg)
                -> decltype(::erf(arg))
                {
                    alpaka::ignore_unused(erf_ctx);
                    return ::erf(arg);
                }
            };
        }
    }
}

#endif
