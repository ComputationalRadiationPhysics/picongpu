/* Copyright 2020 Sergei Bastrakov
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

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/intrinsic/Traits.hpp>

namespace alpaka
{
    //#############################################################################
    //! The GPU CUDA/HIP intrinsic.
    class IntrinsicUniformCudaHipBuiltIn
        : public concepts::Implements<ConceptIntrinsic, IntrinsicUniformCudaHipBuiltIn>
    {
    public:
        //-----------------------------------------------------------------------------
        IntrinsicUniformCudaHipBuiltIn() = default;
        //-----------------------------------------------------------------------------
        __device__ IntrinsicUniformCudaHipBuiltIn(IntrinsicUniformCudaHipBuiltIn const&) = delete;
        //-----------------------------------------------------------------------------
        __device__ IntrinsicUniformCudaHipBuiltIn(IntrinsicUniformCudaHipBuiltIn&&) = delete;
        //-----------------------------------------------------------------------------
        __device__ auto operator=(IntrinsicUniformCudaHipBuiltIn const&) -> IntrinsicUniformCudaHipBuiltIn& = delete;
        //-----------------------------------------------------------------------------
        __device__ auto operator=(IntrinsicUniformCudaHipBuiltIn&&) -> IntrinsicUniformCudaHipBuiltIn& = delete;
        //-----------------------------------------------------------------------------
        ~IntrinsicUniformCudaHipBuiltIn() = default;
    };

    namespace traits
    {
        //#############################################################################
        template<>
        struct Popcount<IntrinsicUniformCudaHipBuiltIn>
        {
            //-----------------------------------------------------------------------------
            __device__ static auto popcount(IntrinsicUniformCudaHipBuiltIn const& /*intrinsic*/, std::uint32_t value)
                -> std::int32_t
            {
#    if BOOST_COMP_CLANG && BOOST_LANG_CUDA
                return __popc(static_cast<int>(value));
#    else
                return __popc(static_cast<unsigned int>(value));
#    endif
            }

            //-----------------------------------------------------------------------------
            __device__ static auto popcount(IntrinsicUniformCudaHipBuiltIn const& /*intrinsic*/, std::uint64_t value)
                -> std::int32_t
            {
#    if BOOST_COMP_CLANG && BOOST_LANG_CUDA
                return __popcll(static_cast<long long>(value));
#    else
                return __popcll(static_cast<unsigned long long>(value));
#    endif
            }
        };

        //#############################################################################
        template<>
        struct Ffs<IntrinsicUniformCudaHipBuiltIn>
        {
            //-----------------------------------------------------------------------------
            __device__ static auto ffs(IntrinsicUniformCudaHipBuiltIn const& /*intrinsic*/, std::int32_t value)
                -> std::int32_t
            {
                return __ffs(static_cast<int>(value));
            }

            //-----------------------------------------------------------------------------
            __device__ static auto ffs(IntrinsicUniformCudaHipBuiltIn const& /*intrinsic*/, std::int64_t value)
                -> std::int32_t
            {
                return __ffsll(static_cast<long long>(value));
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
