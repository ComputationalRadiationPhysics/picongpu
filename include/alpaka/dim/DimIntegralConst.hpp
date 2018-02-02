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

#include <alpaka/dim/Traits.hpp>

#include <type_traits>

namespace alpaka
{
    namespace dim
    {
        //-----------------------------------------------------------------------------
        // N(th) dimension(s).
        template<
            std::size_t N>
        using DimInt = std::integral_constant<std::size_t, N>;

        //-----------------------------------------------------------------------------
        // Trait specializations for integral_constant types.
        /*namespace traits
        {
            //#############################################################################
            //! The arithmetic type dimension getter trait specialization.
            template<
                std::size_t N>
            struct DimType<
                std::integral_constant<std::size_t, N>
            {
                using type = DimInt<N>;
            };
        }*/
    }
}
