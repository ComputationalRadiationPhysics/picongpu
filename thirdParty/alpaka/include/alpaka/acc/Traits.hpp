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

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/core/Common.hpp>

#include <string>
#include <typeinfo>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The accelerator specifics.
    namespace acc
    {
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
        -> AccDevProps<dim::Dim<TAcc>, size::Size<TAcc>>
        {
            return
                traits::GetAccDevProps<
                    TAcc>
                ::getAccDevProps(
                    dev);
        }

        //-----------------------------------------------------------------------------
        //! Writes the accelerator name to the given stream.
        //!
        //! \tparam TAcc The accelerator type to write the name of.
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
}
