/* Copyright 2013-2023 Rene Widera
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

#include "pmacc/attribute/FunctionSpecifier.hpp"

#include <cstdint>


namespace pmacc
{
    template<typename T_Type, typename T_size>
    class StaticArray
    {
    public:
        static constexpr uint32_t size = T_size::value;
        using Type = T_Type;

    private:
        Type data[size];

    public:
        HDINLINE
        Type& operator[](const int idx)
        {
            return data[idx];
        }

        HDINLINE
        const Type& operator[](const int idx) const
        {
            return data[idx];
        }
    };

} // namespace pmacc
