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
#    include <alpaka/math/atan2/Traits.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The CUDA built in atan2.
        class Atan2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2UniformCudaHipBuiltIn>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The CUDA atan2 trait specialization.
            template<typename Ty, typename Tx>
            struct Atan2<
                Atan2UniformCudaHipBuiltIn,
                Ty,
                Tx,
                std::enable_if_t<std::is_floating_point<Ty>::value && std::is_floating_point<Tx>::value>>
            {
                __device__ static auto atan2(Atan2UniformCudaHipBuiltIn const& atan2_ctx, Ty const& y, Tx const& x)
                {
                    alpaka::ignore_unused(atan2_ctx);
                    return ::atan2(y, x);
                }
            };

            template<>
            struct Atan2<Atan2UniformCudaHipBuiltIn, float, float>
            {
                __device__ static auto atan2(
                    Atan2UniformCudaHipBuiltIn const& atan2_ctx,
                    float const& y,
                    float const& x) -> float
                {
                    alpaka::ignore_unused(atan2_ctx);
                    return ::atan2f(y, x);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
