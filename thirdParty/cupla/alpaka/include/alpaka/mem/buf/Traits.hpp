/* Copyright 2019 Alexander Matthes, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/mem/view/Traits.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The buffer traits.
    namespace traits
    {
        //#############################################################################
        //! The memory buffer type trait.
        template<typename TDev, typename TElem, typename TDim, typename TIdx, typename TSfinae = void>
        struct BufType;

        //#############################################################################
        //! The memory allocator trait.
        template<typename TElem, typename TDim, typename TIdx, typename TDev, typename TSfinae = void>
        struct BufAlloc;

        //#############################################################################
        //! The memory mapping trait.
        template<typename TBuf, typename TDev, typename TSfinae = void>
        struct Map;

        //#############################################################################
        //! The memory unmapping trait.
        template<typename TBuf, typename TDev, typename TSfinae = void>
        struct Unmap;

        //#############################################################################
        //! The memory pinning trait.
        template<typename TBuf, typename TSfinae = void>
        struct Pin;

        //#############################################################################
        //! The memory unpinning trait.
        template<typename TBuf, typename TSfinae = void>
        struct Unpin;

        //#############################################################################
        //! The memory pin state trait.
        template<typename TBuf, typename TSfinae = void>
        struct IsPinned;

        //#############################################################################
        //! The memory prepareForAsyncCopy trait.
        template<typename TBuf, typename TSfinae = void>
        struct PrepareForAsyncCopy;
    } // namespace traits

    //#############################################################################
    //! The memory buffer type trait alias template to remove the ::type.
    template<typename TDev, typename TElem, typename TDim, typename TIdx>
    using Buf = typename traits::BufType<alpaka::Dev<TDev>, TElem, TDim, TIdx>::type;

    //-----------------------------------------------------------------------------
    //! Allocates memory on the given device.
    //!
    //! \tparam TElem The element type of the returned buffer.
    //! \tparam TIdx The linear index type of the buffer.
    //! \tparam TExtent The extent type of the buffer.
    //! \tparam TDev The type of device the buffer is allocated on.
    //! \param dev The device to allocate the buffer on.
    //! \param extent The extent of the buffer.
    //! \return The newly allocated buffer.
    template<typename TElem, typename TIdx, typename TExtent, typename TDev>
    ALPAKA_FN_HOST auto allocBuf(TDev const& dev, TExtent const& extent = TExtent())
    {
        return traits::BufAlloc<TElem, Dim<TExtent>, TIdx, TDev>::allocBuf(dev, extent);
    }
    //-----------------------------------------------------------------------------
    //! Maps the buffer into the memory of the given device.
    //!
    //! \tparam TBuf The buffer type.
    //! \tparam TDev The device type.
    //! \param buf The buffer to map into the device memory.
    //! \param dev The device to map the buffer into.
    template<typename TBuf, typename TDev>
    ALPAKA_FN_HOST auto map(TBuf& buf, TDev const& dev) -> void
    {
        return traits::Map<TBuf, TDev>::map(buf, dev);
    }
    //-----------------------------------------------------------------------------
    //! Unmaps the buffer from the memory of the given device.
    //!
    //! \tparam TBuf The buffer type.
    //! \tparam TDev The device type.
    //! \param buf The buffer to unmap from the device memory.
    //! \param dev The device to unmap the buffer from.
    template<typename TBuf, typename TDev>
    ALPAKA_FN_HOST auto unmap(TBuf& buf, TDev const& dev) -> void
    {
        return traits::Unmap<TBuf, TDev>::unmap(buf, dev);
    }
    //-----------------------------------------------------------------------------
    //! Pins the buffer.
    //!
    //! \tparam TBuf The buffer type.
    //! \param buf The buffer to pin in the device memory.
    template<typename TBuf>
    ALPAKA_FN_HOST auto pin(TBuf& buf) -> void
    {
        return traits::Pin<TBuf>::pin(buf);
    }
    //-----------------------------------------------------------------------------
    //! Unpins the buffer.
    //!
    //! \tparam TBuf The buffer type.
    //! \param buf The buffer to unpin from the device memory.
    template<typename TBuf>
    ALPAKA_FN_HOST auto unpin(TBuf& buf) -> void
    {
        return traits::Unpin<TBuf>::unpin(buf);
    }
    //-----------------------------------------------------------------------------
    //! The pin state of the buffer.
    //!
    //! \tparam TBuf The buffer type.
    //! \param buf The buffer to get the pin state of.
    template<typename TBuf>
    ALPAKA_FN_HOST auto isPinned(TBuf const& buf) -> bool
    {
        return traits::IsPinned<TBuf>::isPinned(buf);
    }
    //-----------------------------------------------------------------------------
    //! Prepares the buffer for non-blocking copy operations, e.g. pinning if
    //! non-blocking copy between a cpu and a cuda device is wanted
    //!
    //! \tparam TBuf The buffer type.
    //! \param buf The buffer to prepare in the device memory.
    template<typename TBuf>
    ALPAKA_FN_HOST auto prepareForAsyncCopy(TBuf& buf) -> void
    {
        return traits::PrepareForAsyncCopy<TBuf>::prepareForAsyncCopy(buf);
    }
} // namespace alpaka
