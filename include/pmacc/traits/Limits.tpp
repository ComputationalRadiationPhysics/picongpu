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

#include "pmacc/traits/Limits.hpp"

#include <climits>

namespace pmacc
{
    namespace traits
    {
        namespace limits
        {
            template<>
            struct Max<int>
            {
                static constexpr int value = INT_MAX;
            };

            template<>
            struct Max<uint32_t>
            {
                static constexpr uint32_t value = static_cast<uint32_t>(-1);
            };

            template<>
            struct Max<uint64_t>
            {
                static constexpr uint64_t value = static_cast<uint64_t>(-1);
            };

        } // namespace limits
    } // namespace traits
} // namespace pmacc
