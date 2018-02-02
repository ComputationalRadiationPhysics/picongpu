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
    //! The size specifics.
    namespace size
    {
        //-----------------------------------------------------------------------------
        //! The size traits.
        namespace traits
        {
            //#############################################################################
            //! The size type trait.
            template<
                typename T,
                typename TSfinae = void>
            struct SizeType;
        }

        template<
            typename T>
        using Size = typename traits::SizeType<T>::type;

        //-----------------------------------------------------------------------------
        // Trait specializations for unsigned integral types.
        namespace traits
        {
            //#############################################################################
            //! The arithmetic size type trait specialization.
            template<
                typename T>
            struct SizeType<
                T,
                typename std::enable_if<std::is_arithmetic<T>::value>::type>
            {
                using type = typename std::decay<T>::type;
            };
        }
    }
}
