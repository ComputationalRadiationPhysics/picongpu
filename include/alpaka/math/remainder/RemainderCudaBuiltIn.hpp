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

#include <alpaka/math/remainder/Traits.hpp>

#include <cuda_runtime.h>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library remainder.
        class RemainderCudaBuiltIn
        {
        public:
            using RemainderBase = RemainderCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library remainder trait specialization.
            template<
                typename TArg>
            struct Remainder<
                RemainderCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto remainder(
                    RemainderCudaBuiltIn const & remainder_ctx,
                    TArg const & arg)
                -> decltype(::remainder(arg))
                {
                    alpaka::ignore_unused(remainder_ctx);
                    return ::remainder(arg);
                }
            };
        }
    }
}

#endif
