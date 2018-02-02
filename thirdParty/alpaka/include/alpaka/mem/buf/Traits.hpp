/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/mem/view/Traits.hpp>

#include <alpaka/core/Common.hpp>

#include <boost/config.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The memory specifics.
    namespace mem
    {
        //-----------------------------------------------------------------------------
        //! The buffer specifics.
        namespace buf
        {
            //-----------------------------------------------------------------------------
            //! The buffer traits.
            namespace traits
            {
                //#############################################################################
                //! The memory buffer type trait.
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize,
                    typename TSfinae = void>
                struct BufType;

                //#############################################################################
                //! The memory allocator trait.
                template<
                    typename TElem,
                    typename TDim,
                    typename TSize,
                    typename TDev,
                    typename TSfinae = void>
                struct Alloc;

                //#############################################################################
                //! The memory mapping trait.
                template<
                    typename TBuf,
                    typename TDev,
                    typename TSfinae = void>
                struct Map;

                //#############################################################################
                //! The memory unmapping trait.
                template<
                    typename TBuf,
                    typename TDev,
                    typename TSfinae = void>
                struct Unmap;

                //#############################################################################
                //! The memory pinning trait.
                template<
                    typename TBuf,
                    typename TSfinae = void>
                struct Pin;

                //#############################################################################
                //! The memory unpinning trait.
                template<
                    typename TBuf,
                    typename TSfinae = void>
                struct Unpin;

                //#############################################################################
                //! The memory pin state trait.
                template<
                    typename TBuf,
                    typename TSfinae = void>
                struct IsPinned;
            }

            //#############################################################################
            //! The memory buffer type trait alias template to remove the ::type.
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            using Buf = typename traits::BufType<TDev, TElem, TDim, TSize>::type;

            //-----------------------------------------------------------------------------
            //! Allocates memory on the given device.
            //!
            //! \tparam TElem The element type of the returned buffer.
            //! \tparam TExtent The extent of the buffer.
            //! \tparam TDev The type of device the buffer is allocated on.
            //! \param dev The device to allocate the buffer on.
            //! \param extent The extent of the buffer.
            //! \return The newly allocated buffer.
            template<
                typename TElem,
                typename TSize,
                typename TExtent,
                typename TDev>
            ALPAKA_FN_HOST auto alloc(
                TDev const & dev,
                TExtent const & extent = TExtent())
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                traits::Alloc<
                    TElem,
                    dim::Dim<TExtent>,
                    TSize,
                    TDev>
                ::alloc(
                    dev,
                    extent))
#endif
            {
                return
                    traits::Alloc<
                        TElem,
                        dim::Dim<TExtent>,
                        TSize,
                        TDev>
                    ::alloc(
                        dev,
                        extent);
            }
            //-----------------------------------------------------------------------------
            //! Maps the buffer into the memory of the given device.
            //!
            //! \tparam TBuf The buffer type.
            //! \tparam TDev The device type.
            //! \param buf The buffer to map into the device memory.
            //! \param dev The device to map the buffer into.
            template<
                typename TBuf,
                typename TDev>
            ALPAKA_FN_HOST auto map(
                TBuf & buf,
                TDev const & dev)
            -> void
            {
                return
                    traits::Map<
                        TBuf,
                        TDev>
                    ::map(
                        buf,
                        dev);
            }
            //-----------------------------------------------------------------------------
            //! Unmaps the buffer from the memory of the given device.
            //!
            //! \tparam TBuf The buffer type.
            //! \tparam TDev The device type.
            //! \param buf The buffer to unmap from the device memory.
            //! \param dev The device to unmap the buffer from.
            template<
                typename TBuf,
                typename TDev>
            ALPAKA_FN_HOST auto unmap(
                TBuf & buf,
                TDev const & dev)
            -> void
            {
                return
                    traits::Unmap<
                        TBuf,
                        TDev>
                    ::unmap(
                        buf,
                        dev);
            }
            //-----------------------------------------------------------------------------
            //! Pins the buffer.
            //!
            //! \tparam TBuf The buffer type.
            //! \param buf The buffer to pin in the device memory.
            template<
                typename TBuf>
            ALPAKA_FN_HOST auto pin(
                TBuf & buf)
            -> void
            {
                return
                    traits::Pin<
                        TBuf>
                    ::pin(
                        buf);
            }
            //-----------------------------------------------------------------------------
            //! Unpins the buffer.
            //!
            //! \tparam TBuf The buffer type.
            //! \param buf The buffer to unpin from the device memory.
            template<
                typename TBuf>
            ALPAKA_FN_HOST auto unpin(
                TBuf & buf)
            -> void
            {
                return
                    traits::Unpin<
                        TBuf>
                    ::unpin(
                        buf);
            }
            //-----------------------------------------------------------------------------
            //! The pin state of the buffer.
            //!
            //! \tparam TBuf The buffer type.
            //! \param buf The buffer to get the pin state of.
            template<
                typename TBuf>
            ALPAKA_FN_HOST auto isPinned(
                TBuf const & buf)
            -> bool
            {
                return
                    traits::IsPinned<
                        TBuf>
                    ::isPinned(
                        buf);
            }
        }
    }
}
