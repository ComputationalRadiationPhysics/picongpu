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
#include <alpaka/dev/Traits.hpp>

#include <alpaka/core/Concepts.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/dev/Traits.hpp>

#include <boost/config.hpp>

#include <vector>
#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The platform specifics.
    namespace pltf
    {
        struct ConceptPltf{};

        //-----------------------------------------------------------------------------
        //! The platform traits.
        namespace traits
        {
            //#############################################################################
            //! The platform type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct PltfType;

            //#############################################################################
            //! The device count get trait.
            template<
                typename T,
                typename TSfinae = void>
            struct GetDevCount;

            //#############################################################################
            //! The device get trait.
            template<
                typename T,
                typename TSfinae = void>
            struct GetDevByIdx;
        }

        //#############################################################################
        //! The platform type trait alias template to remove the ::type.
        template<
            typename T>
        using Pltf = typename traits::PltfType<T>::type;

        //-----------------------------------------------------------------------------
        //! \return The device identified by its index.
        template<
            typename TPltf>
        ALPAKA_FN_HOST auto getDevCount()
        {
            return
                traits::GetDevCount<
                    TPltf>
                ::getDevCount();
        }

        //-----------------------------------------------------------------------------
        //! \return The device identified by its index.
        template<
            typename TPltf>
        ALPAKA_FN_HOST auto getDevByIdx(
            std::size_t const & devIdx)
        {
            return
                traits::GetDevByIdx<
                    TPltf>
                ::getDevByIdx(
                    devIdx);
        }

        //-----------------------------------------------------------------------------
        //! \return All the devices available on this accelerator.
        template<
            typename TPltf>
        ALPAKA_FN_HOST auto getDevs()
        -> std::vector<dev::Dev<TPltf>>
        {
            std::vector<dev::Dev<TPltf>> devs;

            std::size_t const devCount(getDevCount<TPltf>());
            for(std::size_t devIdx(0); devIdx < devCount; ++devIdx)
            {
                devs.push_back(getDevByIdx<TPltf>(devIdx));
            }

            return devs;
        }
    }
    namespace queue
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TProperty>
            struct QueueType<
                TPltf,
                TProperty,
                std::enable_if_t<concepts::ImplementsConcept<pltf::ConceptPltf, TPltf>::value>
            >
            {
                using type = typename QueueType<
                    typename dev::traits::DevType<TPltf>::type,
                    TProperty>::type;
            };
        }
    }
}
