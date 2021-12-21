/* Copyright 2019 Benjamin Worpitz
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
#include <alpaka/dev/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

#include <type_traits>
#include <vector>

namespace alpaka
{
    struct ConceptPltf
    {
    };

    //-----------------------------------------------------------------------------
    //! The platform traits.
    namespace traits
    {
        //#############################################################################
        //! The platform type trait.
        template<typename T, typename TSfinae = void>
        struct PltfType;

        template<typename TPltf>
        struct PltfType<TPltf, typename std::enable_if<concepts::ImplementsConcept<ConceptPltf, TPltf>::value>::type>
        {
            using type = typename concepts::ImplementationBase<ConceptDev, TPltf>;
        };

        //#############################################################################
        //! The device count get trait.
        template<typename T, typename TSfinae = void>
        struct GetDevCount;

        //#############################################################################
        //! The device get trait.
        template<typename T, typename TSfinae = void>
        struct GetDevByIdx;
    } // namespace traits

    //#############################################################################
    //! The platform type trait alias template to remove the ::type.
    template<typename T>
    using Pltf = typename traits::PltfType<T>::type;

    //-----------------------------------------------------------------------------
    //! \return The device identified by its index.
    template<typename TPltf>
    ALPAKA_FN_HOST auto getDevCount()
    {
        return traits::GetDevCount<Pltf<TPltf>>::getDevCount();
    }

    //-----------------------------------------------------------------------------
    //! \return The device identified by its index.
    template<typename TPltf>
    ALPAKA_FN_HOST auto getDevByIdx(std::size_t const& devIdx)
    {
        return traits::GetDevByIdx<Pltf<TPltf>>::getDevByIdx(devIdx);
    }

    //-----------------------------------------------------------------------------
    //! \return All the devices available on this accelerator.
    template<typename TPltf>
    ALPAKA_FN_HOST auto getDevs() -> std::vector<Dev<Pltf<TPltf>>>
    {
        std::vector<Dev<Pltf<TPltf>>> devs;

        std::size_t const devCount(getDevCount<Pltf<TPltf>>());
        for(std::size_t devIdx(0); devIdx < devCount; ++devIdx)
        {
            devs.push_back(getDevByIdx<Pltf<TPltf>>(devIdx));
        }

        return devs;
    }

    namespace traits
    {
        template<typename TPltf, typename TProperty>
        struct QueueType<TPltf, TProperty, std::enable_if_t<concepts::ImplementsConcept<ConceptPltf, TPltf>::value>>
        {
            using type = typename QueueType<typename alpaka::traits::DevType<TPltf>::type, TProperty>::type;
        };
    } // namespace traits
} // namespace alpaka
