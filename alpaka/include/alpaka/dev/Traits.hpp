/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace alpaka
{
    //! The device traits.
    namespace trait
    {
        //! The device type trait.
        template<typename T, typename TSfinae = void>
        struct DevType;

        //! The device get trait.
        template<typename T, typename TSfinae = void>
        struct GetDev;

        //! The device name get trait.
        template<typename TDev, typename TSfinae = void>
        struct GetName;

        //! The device memory size get trait.
        template<typename TDev, typename TSfinae = void>
        struct GetMemBytes;

        //! The device free memory size get trait.
        template<typename T, typename TSfinae = void>
        struct GetFreeMemBytes;

        //! The device warp size get trait.
        template<typename T, typename TSfinae = void>
        struct GetWarpSizes;

        //! The device reset trait.
        template<typename T, typename TSfinae = void>
        struct Reset;
    } // namespace trait

    //! The device type trait alias template to remove the ::type.
    template<typename T>
    using Dev = typename trait::DevType<T>::type;

    struct ConceptGetDev;

    struct ConceptDev;

    //! \return The device this object is bound to.
    template<typename T>
    ALPAKA_FN_HOST auto getDev(T const& t)
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptGetDev, T>;
        return trait::GetDev<ImplementationBase>::getDev(t);
    }

    //! \return The device name.
    template<typename TDev>
    ALPAKA_FN_HOST auto getName(TDev const& dev) -> std::string
    {
        return trait::GetName<TDev>::getName(dev);
    }

    //! \return The memory on the device in Bytes. Returns 0 if querying memory
    //!  is not supported.
    template<typename TDev>
    ALPAKA_FN_HOST auto getMemBytes(TDev const& dev) -> std::size_t
    {
        return trait::GetMemBytes<TDev>::getMemBytes(dev);
    }

    //! \return The free memory on the device in Bytes.
    //
    //! \note Do not use this query if getMemBytes returned 0.
    template<typename TDev>
    ALPAKA_FN_HOST auto getFreeMemBytes(TDev const& dev) -> std::size_t
    {
        return trait::GetFreeMemBytes<TDev>::getFreeMemBytes(dev);
    }

    //! \return The supported warp sizes on the device in number of threads.
    template<typename TDev>
    ALPAKA_FN_HOST auto getWarpSizes(TDev const& dev) -> std::vector<std::size_t>
    {
        return trait::GetWarpSizes<TDev>::getWarpSizes(dev);
    }

    //! Resets the device.
    //! What this method does is dependent on the accelerator.
    template<typename TDev>
    ALPAKA_FN_HOST auto reset(TDev const& dev) -> void
    {
        trait::Reset<TDev>::reset(dev);
    }

    namespace trait
    {
        //! Get device type
        template<typename TDev>
        struct DevType<TDev, std::enable_if_t<concepts::ImplementsConcept<ConceptDev, TDev>::value>>
        {
            using type = typename concepts::ImplementationBase<ConceptDev, TDev>;
        };
    } // namespace trait
} // namespace alpaka
