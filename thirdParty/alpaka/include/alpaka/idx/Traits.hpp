/**
* \file
* Copyright 2014-2018 Benjamin Worpitz
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

#include <utility>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The index specifics.
    namespace idx
    {
        //-----------------------------------------------------------------------------
        //! The idx traits.
        namespace traits
        {
            //#############################################################################
            //! The idx type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct IdxType;
        }

        template<
            typename T>
        using Idx = typename traits::IdxType<T>::type;

        //-----------------------------------------------------------------------------
        // Trait specializations for unsigned integral types.
        namespace traits
        {
            //#############################################################################
            //! The arithmetic idx type trait specialization.
            template<
                typename T>
            struct IdxType<
                T,
                typename std::enable_if<std::is_arithmetic<T>::value>::type>
            {
                using type = typename std::decay<T>::type;
            };
        }

        //-----------------------------------------------------------------------------
        //! The index traits.
        namespace traits
        {
            //#############################################################################
            //! The index get trait.
            template<
                typename TIdx,
                typename TOrigin,
                typename TUnit,
                typename TSfinae = void>
            struct GetIdx;
        }
    }
}
