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

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The element specifics.
    namespace elem
    {
        //-----------------------------------------------------------------------------
        //! The element traits.
        namespace traits
        {
            //#############################################################################
            //! The element type trait.
            template<
                typename TView,
                typename TSfinae = void>
            struct ElemType;
        }

        //#############################################################################
        //! The element type trait alias template to remove the ::type.
        template<
            typename TView>
        using Elem = typename std::remove_volatile<typename traits::ElemType<TView>::type>::type;

        //-----------------------------------------------------------------------------
        // Trait specializations for unsigned integral types.
        namespace traits
        {
            //#############################################################################
            //! The fundamental type elem type trait specialization.
            template<
                typename T>
            struct ElemType<
                T,
                typename std::enable_if<std::is_fundamental<T>::value>::type>
            {
                using type = T;
            };
        }
    }
}
