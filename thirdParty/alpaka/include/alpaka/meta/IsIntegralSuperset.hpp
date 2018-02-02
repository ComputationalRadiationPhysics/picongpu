/**
 * \file
 * Copyright 2017 Benjamin Worpitz
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
    namespace meta
    {
        //#############################################################################
        //! The trait is true if all values of TSubset are contained in TSuperset.
        template<
            typename TSuperset,
            typename TSubset>
        using IsIntegralSuperset =
            std::integral_constant<
                bool,
                std::is_integral<TSuperset>::value && std::is_integral<TSubset>::value
                && (
                    // If the signdness is equal, the sizes have to be greater or equal to be a superset.
                    ((std::is_unsigned<TSuperset>::value == std::is_unsigned<TSubset>::value) && (sizeof(TSuperset) >= sizeof(TSubset)))
                    // If the signdness is non-equal, the superset has to have at least one bit more.
                    || ((std::is_unsigned<TSuperset>::value != std::is_unsigned<TSubset>::value) && (sizeof(TSuperset) > sizeof(TSubset)))
                )>;
    }
}
