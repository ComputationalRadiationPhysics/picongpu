/* Copyright 2022 Sergei Bastrakov, David M. Rogers, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/warp/Traits.hpp>

#    include <cstdint>

namespace alpaka::warp
{
    //! The GPU CUDA/HIP warp.
    class WarpUniformCudaHipBuiltIn : public concepts::Implements<ConceptWarp, WarpUniformCudaHipBuiltIn>
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace trait
    {
        template<>
        struct GetSize<WarpUniformCudaHipBuiltIn>
        {
            __device__ static auto getSize(warp::WarpUniformCudaHipBuiltIn const& /*warp*/) -> std::int32_t
            {
                return warpSize;
            }
        };

        template<>
        struct Activemask<WarpUniformCudaHipBuiltIn>
        {
            __device__ static auto activemask(warp::WarpUniformCudaHipBuiltIn const& /*warp*/)
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                -> std::uint32_t
#        else
                -> std::uint64_t
#        endif
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __activemask();
#        else
                // No HIP intrinsic for it, emulate via ballot
                return __ballot(1);
#        endif
            }
        };

        template<>
        struct All<WarpUniformCudaHipBuiltIn>
        {
            __device__ static auto all(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                std::int32_t predicate) -> std::int32_t
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __all_sync(activemask(warp), predicate);
#        else
                return __all(predicate);
#        endif
            }
        };

        template<>
        struct Any<WarpUniformCudaHipBuiltIn>
        {
            __device__ static auto any(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                std::int32_t predicate) -> std::int32_t
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __any_sync(activemask(warp), predicate);
#        else
                return __any(predicate);
#        endif
            }
        };

        template<>
        struct Ballot<WarpUniformCudaHipBuiltIn>
        {
            __device__ static auto ballot(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                std::int32_t predicate)
            // return type is required by the compiler
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                -> std::uint32_t
#        else
                -> std::uint64_t
#        endif
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __ballot_sync(activemask(warp), predicate);
#        else
                return __ballot(predicate);
#        endif
            }
        };

        template<>
        struct Shfl<WarpUniformCudaHipBuiltIn>
        {
            //-------------------------------------------------------------
            __device__ static auto shfl(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                float val,
                int srcLane,
                std::int32_t width) -> float
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __shfl_sync(activemask(warp), val, srcLane, width);
#        else
                return __shfl(val, srcLane, width);
#        endif
            }
            //-------------------------------------------------------------
            __device__ static auto shfl(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                std::int32_t val,
                int srcLane,
                std::int32_t width) -> std::int32_t
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __shfl_sync(activemask(warp), val, srcLane, width);
#        else
                return __shfl(val, srcLane, width);
#        endif
            }
        };
    } // namespace trait
#    endif
} // namespace alpaka::warp

#endif
