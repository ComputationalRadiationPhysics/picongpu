/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/core/Common.hpp>

#include <alpaka/core/Concepts.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>

#include <string>
#include <typeinfo>
#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The accelerator specifics.
    namespace acc
    {
        struct ConceptAcc;

        //-----------------------------------------------------------------------------
        //! The accelerator traits.
        namespace traits
        {
            //#############################################################################
            //! The accelerator type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct AccType;

            //#############################################################################
            //! The device properties get trait.
            template<
                typename TAcc,
                typename TSfinae = void>
            struct GetAccDevProps;

            //#############################################################################
            //! The accelerator name trait.
            //!
            //! The default implementation returns the mangled class name.
            template<
                typename TAcc,
                typename TSfinae = void>
            struct GetAccName
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return typeid(TAcc).name();
                }
            };
        }

        //#############################################################################
        //! The accelerator type trait alias template to remove the ::type.
        template<
            typename T>
        using Acc = typename traits::AccType<T>::type;

        //-----------------------------------------------------------------------------
        //! \return The acceleration properties on the given device.
        template<
            typename TAcc,
            typename TDev>
        ALPAKA_FN_HOST auto getAccDevProps(
            TDev const & dev)
        -> AccDevProps<dim::Dim<TAcc>, idx::Idx<TAcc>>
        {
            return
                traits::GetAccDevProps<
                    TAcc>
                ::getAccDevProps(
                    dev);
        }

        //-----------------------------------------------------------------------------
        //! \return The accelerator name
        //!
        //! \tparam TAcc The accelerator type.
        template<
            typename TAcc>
        ALPAKA_FN_HOST auto getAccName()
        -> std::string
        {
            return
                traits::GetAccName<
                    TAcc>
                ::getAccName();
        }
    }

    namespace queue
    {
        namespace traits
        {
            template<
                typename TAcc,
                typename TProperty>
            struct QueueType<
                TAcc,
                TProperty,
                typename std::enable_if<
                    concepts::ImplementsConcept<acc::ConceptAcc, TAcc>::value
                >::type
            >
            {
                using type = typename QueueType<
                    typename pltf::traits::PltfType<TAcc>::type,
                    TProperty
                >::type;
            };
        }
    }
}
