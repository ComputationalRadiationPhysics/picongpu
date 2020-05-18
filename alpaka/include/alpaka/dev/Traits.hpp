/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>

#include <boost/config.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The device specifics.
    namespace dev
    {
        //-----------------------------------------------------------------------------
        //! The device traits.
        namespace traits
        {
            //#############################################################################
            //! The device type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct DevType;

            //#############################################################################
            //! The device get trait.
            template<
                typename T,
                typename TSfinae = void>
            struct GetDev;

            //#############################################################################
            //! The device name get trait.
            template<
                typename TDev,
                typename TSfinae = void>
            struct GetName;

            //#############################################################################
            //! The device memory size get trait.
            template<
                typename TDev,
                typename TSfinae = void>
            struct GetMemBytes;

            //#############################################################################
            //! The device free memory size get trait.
            template<
                typename T,
                typename TSfinae = void>
            struct GetFreeMemBytes;

            //#############################################################################
            //! The device reset trait.
            template<
                typename T,
                typename TSfinae = void>
            struct Reset;
        }

        //#############################################################################
        //! The device type trait alias template to remove the ::type.
        template<
            typename T>
        using Dev = typename traits::DevType<T>::type;

        //-----------------------------------------------------------------------------
        //! \return The device this object is bound to.
        template<
            typename T>
        ALPAKA_FN_HOST auto getDev(
            T const & t)
        {
            return
                traits::GetDev<
                    T>
                ::getDev(
                    t);
        }

        //-----------------------------------------------------------------------------
        //! \return The device name.
        template<
            typename TDev>
        ALPAKA_FN_HOST auto getName(
            TDev const & dev)
        -> std::string
        {
            return
                traits::GetName<
                    TDev>
                ::getName(
                    dev);
        }

        //-----------------------------------------------------------------------------
        //! \return The memory on the device in Bytes.
        template<
            typename TDev>
        ALPAKA_FN_HOST auto getMemBytes(
            TDev const & dev)
        -> std::size_t
        {
            return
                traits::GetMemBytes<
                    TDev>
                ::getMemBytes(
                    dev);
        }

        //-----------------------------------------------------------------------------
        //! \return The free memory on the device in Bytes.
        template<
            typename TDev>
        ALPAKA_FN_HOST auto getFreeMemBytes(
            TDev const & dev)
        -> std::size_t
        {
            return
                traits::GetFreeMemBytes<
                    TDev>
                ::getFreeMemBytes(
                    dev);
        }

        //-----------------------------------------------------------------------------
        //! Resets the device.
        //! What this method does is dependent on the accelerator.
        template<
            typename TDev>
        ALPAKA_FN_HOST auto reset(
            TDev const & dev)
        -> void
        {
            traits::Reset<
                TDev>
            ::reset(
                dev);
        }
    }
}
