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

#include <alpaka/math/log/Traits.hpp>

#include <alpaka/core/Unused.hpp>

#include <cuda_runtime.h>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        // ! The CUDA built in log.
        class LogCudaBuiltIn : public concepts::Implements<ConceptMathLog, LogCudaBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA log trait specialization.
            template<
                typename TArg>
            struct Log<
                LogCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto log(
                    LogCudaBuiltIn const & log_ctx,
                    TArg const & arg)
                -> decltype(::log(arg))
                {
                    alpaka::ignore_unused(log_ctx);
                    return ::log(arg);
                }
            };
            //! The CUDA log float specialization.
            template<>
            struct Log<
                LogCudaBuiltIn,
                float>
            {
                __device__ static auto log(
                    LogCudaBuiltIn const & log_ctx,
                    float const & arg)
                -> float
                {
                    alpaka::ignore_unused(log_ctx);
                    return ::logf(arg);
                }
            };

        }
    }
}

#endif
