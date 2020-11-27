/* Copyright 2019 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Bert Wesarg
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <cuda_runtime.h>
#        if !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#        if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
#            include <cuda_runtime_api.h>
#        else
#            if BOOST_COMP_HIP
#                include <hip/math_functions.h>
#            else
#                include <math_functions.hpp>
#            endif
#        endif

#        if !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif
#    endif

#    include <alpaka/core/Unused.hpp>
#    include <alpaka/math/min/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in min.
        class MinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMin, MinUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA integral min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<
                MinUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<std::is_integral<Tx>::value && std::is_integral<Ty>::value>>
            {
                __device__ static auto min(MinUniformCudaHipBuiltIn const& min_ctx, Tx const& x, Ty const& y)
                    -> decltype(::min(x, y))
                {
                    alpaka::ignore_unused(min_ctx);
                    return ::min(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<
                MinUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_arithmetic<Tx>::value && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value && std::is_integral<Ty>::value)>>
            {
                __device__ static auto min(MinUniformCudaHipBuiltIn const& min_ctx, Tx const& x, Ty const& y)
                    -> decltype(::fmin(x, y))
                {
                    alpaka::ignore_unused(min_ctx);
                    return ::fmin(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
