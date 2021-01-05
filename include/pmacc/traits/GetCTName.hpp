/* Copyright 2018-2021 Rene Widera
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

#include <pmacc/meta/String.hpp>


namespace pmacc
{
    namespace traits
    {
        /** Return the compile time name
         *
         * @tparam T_Type type of the object where the name is queried
         * @return ::type name of the object as pmacc::meta::String,
         *         empty string is returned if the trait is not specified for
         *         T_Type
         */
        template<typename T_Type>
        struct GetCTName
        {
            using type = pmacc::meta::String<>;
        };

        template<typename T_Type>
        using GetCTName_t = typename GetCTName<T_Type>::type;

    } // namespace traits
} // namespace pmacc
