/* Copyright 2023 Sergei Bastrakov, David M. Rogers, Jan Stephan, Andrea Bocci, Bernhard Manfred Gruber, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/warp/Traits.hpp"

#include <cstdint>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

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
                return __all_sync(0xffff'ffff, predicate);
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
                return __any_sync(0xffff'ffff, predicate);
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
                return __ballot_sync(0xffff'ffff, predicate);
#        else
                return __ballot(predicate);
#        endif
            }
        };

        template<>
        struct Shfl<WarpUniformCudaHipBuiltIn>
        {
            template<typename T>
            __device__ static auto shfl(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                T val,
                int srcLane,
                std::int32_t width) -> T
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __shfl_sync(0xffff'ffff, val, srcLane, width);
#        else
                return __shfl(val, srcLane, width);
#        endif
            }
        };

        template<>
        struct ShflUp<WarpUniformCudaHipBuiltIn>
        {
            template<typename T>
            __device__ static auto shfl_up(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                T val,
                std::uint32_t offset,
                std::int32_t width) -> T
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __shfl_up_sync(0xffff'ffff, val, offset, width);
#        else
                return __shfl_up(val, offset, width);
#        endif
            }
        };

        template<>
        struct ShflDown<WarpUniformCudaHipBuiltIn>
        {
            template<typename T>
            __device__ static auto shfl_down(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                T val,
                std::uint32_t offset,
                std::int32_t width) -> T
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __shfl_down_sync(0xffff'ffff, val, offset, width);
#        else
                return __shfl_down(val, offset, width);
#        endif
            }
        };

        template<>
        struct ShflXor<WarpUniformCudaHipBuiltIn>
        {
            template<typename T>
            __device__ static auto shfl_xor(
                [[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
                T val,
                std::int32_t mask,
                std::int32_t width) -> T
            {
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                return __shfl_xor_sync(0xffff'ffff, val, mask, width);
#        else
                return __shfl_xor(val, mask, width);
#        endif
            }
        };

    } // namespace trait
#    endif
} // namespace alpaka::warp

#endif
