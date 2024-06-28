/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/mem/global/Traits.hpp"
#include "alpaka/mem/view/ViewPlainPtr.hpp"
#include "alpaka/queue/cuda_hip/QueueUniformCudaHipRt.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{

    namespace detail
    {
        template<typename T>
        struct DevGlobalTrait<TagGpuCudaRt, T>
        {
            // CUDA implementation
            using Type = detail::DevGlobalImplGeneric<TagGpuCudaRt, T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagGpuHipRt, T>
        {
            // HIP/ROCm implementation
            using Type = detail::DevGlobalImplGeneric<TagGpuHipRt, T>;
        };
    } // namespace detail

    // from device to host
    template<
        typename TTag,
        typename TApi,
        bool TBlocking,
        typename TViewDst,
        typename TTypeSrc,
        typename std::enable_if_t<
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            (std::is_same_v<TTag, TagGpuCudaRt> && std::is_same_v<TApi, ApiCudaRt>)
#    else
            (std::is_same_v<TTag, TagGpuHipRt> && std::is_same_v<TApi, ApiHipRt>)
#    endif
                ,
            int>
        = 0>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        TViewDst& viewDst,
        alpaka::detail::DevGlobalImplGeneric<TTag, TTypeSrc>& viewSrc)
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TTypeSrc>>;
        using TypeExt = std::remove_const_t<TTypeSrc>;
        auto extent = getExtents(viewDst);
        TypeExt* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(const_cast<TypeExt*>(&viewSrc))));

        auto view = alpaka::ViewPlainPtr<
            DevUniformCudaHipRt<TApi>,
            Type,
            alpaka::Dim<decltype(extent)>,
            alpaka::Idx<decltype(extent)>>(reinterpret_cast<Type*>(pMemAcc), alpaka::getDev(queue), extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDst>(viewDst), view, extent));
    }

    // from host to device
    template<
        typename TTag,
        typename TApi,
        bool TBlocking,
        typename TTypeDst,
        typename TViewSrc,
        typename std::enable_if_t<
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            (std::is_same_v<TTag, TagGpuCudaRt> && std::is_same_v<TApi, ApiCudaRt>)
#    else
            (std::is_same_v<TTag, TagGpuHipRt> && std::is_same_v<TApi, ApiHipRt>)
#    endif
                ,
            int>
        = 0>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        alpaka::detail::DevGlobalImplGeneric<TTag, TTypeDst>& viewDst,
        TViewSrc const& viewSrc)
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TTypeDst>>;
        using TypeExt = std::remove_const_t<TTypeDst>;
        auto extent = getExtents(viewSrc);
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(const_cast<TypeExt*>(&viewDst))));

        auto view = alpaka::ViewPlainPtr<
            DevUniformCudaHipRt<TApi>,
            Type,
            alpaka::Dim<decltype(extent)>,
            alpaka::Idx<decltype(extent)>>(reinterpret_cast<Type*>(pMemAcc), alpaka::getDev(queue), extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    // from device to host
    template<
        typename TTag,
        typename TApi,
        bool TBlocking,
        typename TViewDst,
        typename TTypeSrc,
        typename TExtent,
        typename std::enable_if_t<
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            (std::is_same_v<TTag, TagGpuCudaRt> && std::is_same_v<TApi, ApiCudaRt>)
#    else
            (std::is_same_v<TTag, TagGpuHipRt> && std::is_same_v<TApi, ApiHipRt>)
#    endif
                ,
            int>
        = 0>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        TViewDst& viewDst,
        alpaka::detail::DevGlobalImplGeneric<TTag, TTypeSrc>& viewSrc,
        TExtent extent)
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TTypeSrc>>;
        using TypeExt = std::remove_const_t<TTypeSrc>;
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(const_cast<TypeExt*>(&viewSrc))));

        auto view = alpaka::ViewPlainPtr<DevUniformCudaHipRt<TApi>, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<Type*>(pMemAcc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDst>(viewDst), view, extent));
    }

    // from host to device
    template<
        typename TTag,
        typename TApi,
        bool TBlocking,
        typename TTypeDst,
        typename TViewSrc,
        typename TExtent,
        typename std::enable_if_t<
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
            (std::is_same_v<TTag, TagGpuCudaRt> && std::is_same_v<TApi, ApiCudaRt>)
#    else
            (std::is_same_v<TTag, TagGpuHipRt> && std::is_same_v<TApi, ApiHipRt>)
#    endif
                ,
            int>
        = 0>
    ALPAKA_FN_HOST auto memcpy(
        uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
        alpaka::detail::DevGlobalImplGeneric<TTag, TTypeDst>& viewDst,
        TViewSrc const& viewSrc,
        TExtent extent)
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TTypeDst>>;
        using TypeExt = std::remove_const_t<TTypeDst>;
        Type* pMemAcc(nullptr);
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *(const_cast<TypeExt*>(&viewDst))));

        auto view = alpaka::ViewPlainPtr<DevUniformCudaHipRt<TApi>, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<Type*>(pMemAcc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }
} // namespace alpaka

#endif
