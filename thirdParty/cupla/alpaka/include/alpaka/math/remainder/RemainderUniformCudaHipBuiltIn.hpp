/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg
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
#    include <alpaka/math/remainder/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in remainder.
        class RemainderUniformCudaHipBuiltIn
            : public concepts::Implements<ConceptMathRemainder, RemainderUniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA remainder trait specialization.
            template<typename Tx, typename Ty>
            struct Remainder<
                RemainderUniformCudaHipBuiltIn,
                Tx,
                Ty,
                std::enable_if_t<std::is_floating_point<Tx>::value && std::is_floating_point<Ty>::value>>
            {
                __device__ static auto remainder(
                    RemainderUniformCudaHipBuiltIn const& remainder_ctx,
                    Tx const& x,
                    Ty const& y)
                {
                    alpaka::ignore_unused(remainder_ctx);
                    return ::remainder(x, y);
                }
            };
            //! The CUDA remainder float specialization.
            template<>
            struct Remainder<RemainderUniformCudaHipBuiltIn, float, float>
            {
                __device__ static auto remainder(
                    RemainderUniformCudaHipBuiltIn const& remainder_ctx,
                    float const& x,
                    float const& y) -> float
                {
                    alpaka::ignore_unused(remainder_ctx);
                    return ::remainderf(x, y);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
