/* Copyright 2014-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace traits
    {
        /** Get number of possible exchanges
         *
         * \tparam T_dim dimension of the simulation
         * \return \p ::value number of possible exchanges
         *              (is number neighbors + myself)
         */
        template<uint32_t T_dim>
        struct NumberOfExchanges;

        template<>
        struct NumberOfExchanges<DIM1>
        {
            static constexpr uint32_t value = LEFT + RIGHT;
        };

        template<>
        struct NumberOfExchanges<DIM2>
        {
            static constexpr uint32_t value = TOP + BOTTOM;
        };

        template<>
        struct NumberOfExchanges<DIM3>
        {
            static constexpr uint32_t value = BACK + FRONT;
        };

    } // namespace traits

} // namespace pmacc
