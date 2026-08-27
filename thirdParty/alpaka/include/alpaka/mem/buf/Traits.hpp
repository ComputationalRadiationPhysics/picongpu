/* Copyright 2025 Alexander Matthes, Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan,
 *                Christian Kaever, Maria Michailidi, Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Config.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/platform/Traits.hpp"

namespace alpaka
{
    //! The CPU device handle.
    class DevCpu;

    //! The buffer traits.
    namespace trait
    {
        //! The memory buffer type trait.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TSfinae = void>
        struct BufType;

        //! The memory const-buffer type trait.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TSfinae = void>
        struct ConstBufType;

        //! The memory allocator trait.
        template<typename TElem, typename TDim, typename TIdx, typename TDev, typename TSfinae = void>
        struct BufAlloc;

        //! The stream-ordered memory allocator trait.
        template<typename TElem, typename TDim, typename TIdx, typename TDev, typename TSfinae = void>
        struct AsyncBufAlloc;

        //! The stream-ordered memory allocation capability trait.
        template<typename TDim, typename TDev>
        struct HasAsyncBufSupport : public std::false_type
        {
        };

        //! The pinned/mapped memory allocator trait.
        template<typename TPlatform, typename TElem, typename TDim, typename TIdx>
        struct BufAllocMapped;

        //! The pinned/mapped memory allocation capability trait.
        template<typename TPlatform>
        struct HasMappedBufSupport : public std::false_type
        {
        };

        //! The managed (unified) memory allocator trait.
        template<typename TPlatform, typename TElem, typename TDim, typename TIdx>
        struct BufAllocManaged;

        //! The trait to transform a mutable buffer into a constant one.
        template<typename TBuf>
        struct MakeConstBuf;

    } // namespace trait

    //! The memory buffer type trait alias template to remove the ::type for a Buffer type.
    template<typename TDev, typename TElem, typename TDim, typename TIdx>
    using Buf = typename trait::BufType<alpaka::Dev<TDev>, TElem, TDim, TIdx>::type;

    //! The memory buffer type trait alias template to remove the ::type for a ConstBuffer type.
    template<typename TDev, typename TElem, typename TDim, typename TIdx>
    using ConstBuf = typename trait::ConstBufType<alpaka::Dev<TDev>, TElem, TDim, TIdx>::type;

    //! Allocates memory on the given device.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TDev The type of device the buffer is allocated on.
    //! \param dev The device to allocate the buffer on.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx = void, typename TExtent = void, typename TDev = void>
    ALPAKA_FN_HOST auto allocBuf(TDev const& dev, TExtent const& extent = TExtent())
    {
        using Idx = std::conditional_t<std::is_void_v<TIdx>, Idx<TExtent>, TIdx>;

        return trait::BufAlloc<TElem, Dim<TExtent>, Idx, TDev>::allocBuf(dev, extent);
    }

    //! Allocates stream-ordered memory on the given device.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TQueue The type of queue used to order the buffer allocation.
    //! \param queue The queue used to order the buffer allocation.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx = void, typename TExtent = void, typename TQueue = void>
    ALPAKA_FN_HOST auto allocAsyncBuf(TQueue queue, TExtent const& extent = TExtent())
    {
        using Idx = std::conditional_t<std::is_void_v<TIdx>, Idx<TExtent>, TIdx>;

        return trait::AsyncBufAlloc<TElem, Dim<TExtent>, Idx, alpaka::Dev<TQueue>>::allocAsyncBuf(queue, extent);
    }

    /* TODO: Remove this pragma block once support for clang versions <= 13 is removed. These versions are unable to
       figure out that the template parameters are attached to a C++17 inline variable. */
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif
    //! Checks if the given device can allocate a stream-ordered memory buffer of the given dimensionality.
    //!
    //! \tparam TDev The type of device to allocate the buffer on.
    //! \tparam TDim The dimensionality of the buffer to allocate.
    template<typename TDev, typename TDim>
    inline constexpr bool hasAsyncBufSupport = trait::HasAsyncBufSupport<TDim, TDev>::value;
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    //! If supported, allocates stream-ordered memory on the given queue and the associated device.
    //! Otherwise, allocates regular memory on the device associated to the queue.
    //! Please note that stream-ordered and regular memory have different semantics:
    //! this function is provided for convenience in the cases where the difference is not relevant,
    //! and the stream-ordered memory is only used as a performance optimisation.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TQueue The type of queue used to order the buffer allocation.
    //! \param queue The queue used to order the buffer allocation.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx = void, typename TExtent = void, typename TQueue = void>
    ALPAKA_FN_HOST auto allocAsyncBufIfSupported(TQueue queue, TExtent const& extent = TExtent())
    {
        using Idx = std::conditional_t<std::is_void_v<TIdx>, Idx<TExtent>, TIdx>;

        if constexpr(hasAsyncBufSupport<alpaka::Dev<TQueue>, Dim<TExtent>>)
        {
            return allocAsyncBuf<TElem, Idx>(queue, extent);
        }
        else
        {
            return allocBuf<TElem, Idx>(getDev(queue), extent);
        }

        ALPAKA_UNREACHABLE(allocBuf<TElem, TIdx>(getDev(queue), extent));
    }

    //! Allocates pinned/mapped host memory, accessible by all devices in the given platform.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TPlatform The platform from which the buffer is accessible.
    //! \param host The host device to allocate the buffer on.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx = void, typename TExtent = void, typename TPlatform = void>
    ALPAKA_FN_HOST auto allocMappedBuf(
        DevCpu const& host,
        TPlatform const& platform,
        TExtent const& extent = TExtent())
    {
        using Idx = std::conditional_t<std::is_void_v<TIdx>, Idx<TExtent>, TIdx>;

        return trait::BufAllocMapped<TPlatform, TElem, Dim<TExtent>, Idx>::allocMappedBuf(host, platform, extent);
    }

    //! Allocates unified/managed memory, accessible by all devices in the given platform.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TPlatform The platform from which the buffer is accessible.
    //! \param host The host device to allocate the buffer on.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx = void, typename TExtent = void, typename TPlatform = void>
    ALPAKA_FN_HOST auto allocManagedBuf(
        DevCpu const& host,
        TPlatform const& platform,
        TExtent const& extent = TExtent())
    {
        using Idx = std::conditional_t<std::is_void_v<TIdx>, Idx<TExtent>, TIdx>;

        return trait::BufAllocManaged<TPlatform, TElem, Dim<TExtent>, Idx>::allocManagedBuf(host, platform, extent);
    }

    /* TODO: Remove this pragma block once support for clang versions <= 13 is removed. These versions are unable to
       figure out that the template parameters are attached to a C++17 inline variable. */
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif
    //! Checks if the host can allocate a pinned/mapped host memory, accessible by all devices in the given platform.
    //!
    //! \tparam TPlatform The platform from which the buffer is accessible.
    template<typename TPlatform>
    inline constexpr bool hasMappedBufSupport = trait::HasMappedBufSupport<TPlatform>::value;
#if ALPAKA_COMP_CLANG
#    pragma clang diagnostic pop
#endif

    //! If supported, allocates pinned/mapped host memory, accessible by all devices in the given platform.
    //! Otherwise, allocates regular host memory.
    //! Please note that pinned/mapped and regular memory may have different semantics:
    //! this function is provided for convenience in the cases where the difference is not relevant,
    //! and the pinned/mapped memory is only used as a performance optimisation.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TPlatform The platform from which the buffer is accessible.
    //! \param host The host device to allocate the buffer on.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx = void, typename TExtent = void, typename TPlatform = void>
    ALPAKA_FN_HOST auto allocMappedBufIfSupported(
        DevCpu const& host,
        TPlatform const& platform,
        TExtent const& extent = TExtent())
    {
        using Idx = std::conditional_t<std::is_void_v<TIdx>, Idx<TExtent>, TIdx>;
        using Platform = alpaka::Platform<TPlatform>;

        if constexpr(hasMappedBufSupport<Platform>)
        {
            return allocMappedBuf<TElem, Idx>(host, platform, extent);
        }
        else
        {
            return allocBuf<TElem, Idx>(host, extent);
        }

        ALPAKA_UNREACHABLE(allocBuf<TElem, TIdx>(host, extent));
    }

    //! Creates a constant buffer from the given mutable buffer.
    //!
    //! \tparam TBuf The type of the original buffer.
    //! \param buf The original buffer.
    //! \return The transformed buffer with only read-access allowed.
    template<typename TBuf>
    ALPAKA_FN_HOST auto makeConstBuf(TBuf const& buf)
    {
        return trait::MakeConstBuf<TBuf>::makeConstBuf(buf);
    }

    template<typename TBuf>
    ALPAKA_FN_HOST auto makeConstBuf(TBuf&& buf)
    {
        return trait::MakeConstBuf<std::remove_cvref_t<TBuf>>::makeConstBuf(std::move(buf));
    }

} // namespace alpaka
