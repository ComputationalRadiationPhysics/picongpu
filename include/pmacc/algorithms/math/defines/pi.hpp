/* Copyright 2018-2021 Sergei Bastrakov
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

namespace pmacc
{
    namespace math
    {
        /** Values of pi and related constants as T_Type
         */
        template<typename T_Type>
        struct Pi
        {
            static constexpr T_Type value = static_cast<T_Type>(3.141592653589793238462643383279502884197169399);
            static constexpr T_Type doubleValue = static_cast<T_Type>(2.0) * value;
            static constexpr T_Type halfValue = value / static_cast<T_Type>(2.0);
            static constexpr T_Type quarterValue = value / static_cast<T_Type>(4.0);
            static constexpr T_Type doubleReciprocalValue = static_cast<T_Type>(2.0) / value;
        };

    } // namespace math
} // namespace pmacc
