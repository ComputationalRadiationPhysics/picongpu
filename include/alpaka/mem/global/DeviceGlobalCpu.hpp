/* Copyright 2024 Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/buf/cpu/Copy.hpp"
#include "alpaka/mem/global/Traits.hpp"
#include "alpaka/mem/view/ViewPlainPtr.hpp"

#include <type_traits>

// memcpy specialization for device global variables
namespace alpaka
{

    namespace detail
    {
        template<typename T>
        struct DevGlobalTrait<TagCpuOmp2Blocks, T>
        {
            using Type = detail::DevGlobalImplGeneric<TagCpuOmp2Blocks, T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagCpuOmp2Threads, T>
        {
            using Type = detail::DevGlobalImplGeneric<TagCpuOmp2Threads, T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagCpuSerial, T>
        {
            using Type = detail::DevGlobalImplGeneric<TagCpuSerial, T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagCpuTbbBlocks, T>
        {
            using Type = detail::DevGlobalImplGeneric<TagCpuTbbBlocks, T>;
        };

        template<typename T>
        struct DevGlobalTrait<TagCpuThreads, T>
        {
            using Type = detail::DevGlobalImplGeneric<TagCpuThreads, T>;
        };
    } // namespace detail

    template<
        typename TTag,
        typename TViewSrc,
        typename TTypeDst,
        typename TQueue,
        typename std::enable_if_t<
            std::is_same_v<TTag, TagCpuOmp2Blocks> || std::is_same_v<TTag, TagCpuOmp2Threads>
                || std::is_same_v<TTag, TagCpuSerial> || std::is_same_v<TTag, TagCpuTbbBlocks>
                || std::is_same_v<TTag, TagCpuThreads>,
            int>
        = 0>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        alpaka::detail::DevGlobalImplGeneric<TTag, TTypeDst>& viewDst,
        TViewSrc const& viewSrc) -> void
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TTypeDst>>;
        auto extent = getExtents(viewSrc);
        auto view = alpaka::ViewPlainPtr<DevCpu, Type, alpaka::Dim<decltype(extent)>, alpaka::Idx<decltype(extent)>>(
            reinterpret_cast<Type*>(const_cast<std::remove_const_t<TTypeDst>*>(&viewDst)),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    template<
        typename TTag,
        typename TTypeSrc,
        typename TViewDstFwd,
        typename TQueue,
        typename std::enable_if_t<
            std::is_same_v<TTag, TagCpuOmp2Blocks> || std::is_same_v<TTag, TagCpuOmp2Threads>
                || std::is_same_v<TTag, TagCpuSerial> || std::is_same_v<TTag, TagCpuTbbBlocks>
                || std::is_same_v<TTag, TagCpuThreads>,
            int>
        = 0>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        TViewDstFwd&& viewDst,
        alpaka::detail::DevGlobalImplGeneric<TTag, TTypeSrc>& viewSrc) -> void
    {
        using Type = std::remove_all_extents_t<TTypeSrc>;
        auto extent = getExtents(viewDst);
        auto view = alpaka::ViewPlainPtr<DevCpu, Type, alpaka::Dim<decltype(extent)>, alpaka::Idx<decltype(extent)>>(
            reinterpret_cast<Type*>(&viewSrc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), view, extent));
    }

    template<
        typename TTag,
        typename TExtent,
        typename TViewSrc,
        typename TTypeDst,
        typename TQueue,
        typename std::enable_if_t<
            std::is_same_v<TTag, TagCpuOmp2Blocks> || std::is_same_v<TTag, TagCpuOmp2Threads>
                || std::is_same_v<TTag, TagCpuSerial> || std::is_same_v<TTag, TagCpuTbbBlocks>
                || std::is_same_v<TTag, TagCpuThreads>,
            int>
        = 0>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        alpaka::detail::DevGlobalImplGeneric<TTag, TTypeDst>& viewDst,
        TViewSrc const& viewSrc,
        TExtent const& extent) -> void
    {
        using Type = std::remove_const_t<std::remove_all_extents_t<TTypeDst>>;
        auto view = alpaka::ViewPlainPtr<DevCpu, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<Type*>(const_cast<std::remove_const_t<TTypeDst>*>(&viewDst)),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<decltype(view)>(view), viewSrc, extent));
    }

    template<
        typename TTag,
        typename TExtent,
        typename TTypeSrc,
        typename TViewDstFwd,
        typename TQueue,
        typename std::enable_if_t<
            std::is_same_v<TTag, TagCpuOmp2Blocks> || std::is_same_v<TTag, TagCpuOmp2Threads>
                || std::is_same_v<TTag, TagCpuSerial> || std::is_same_v<TTag, TagCpuTbbBlocks>
                || std::is_same_v<TTag, TagCpuThreads>,
            int>
        = 0>
    ALPAKA_FN_HOST auto memcpy(
        TQueue& queue,
        TViewDstFwd&& viewDst,
        alpaka::detail::DevGlobalImplGeneric<TTag, TTypeSrc>& viewSrc,
        TExtent const& extent) -> void
    {
        using Type = std::remove_all_extents_t<TTypeSrc>;
        auto view = alpaka::ViewPlainPtr<DevCpu, Type, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
            reinterpret_cast<Type*>(&viewSrc),
            alpaka::getDev(queue),
            extent);
        enqueue(queue, createTaskMemcpy(std::forward<TViewDstFwd>(viewDst), view, extent));
    }
} // namespace alpaka
