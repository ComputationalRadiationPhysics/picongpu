/* Copyright 2016-2021 Heiko Burau
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

#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace traits
    {
        /** Return an initialized instance. Expects a single parameter.
         *
         * The main reason to use this is for templated types where it's unknown
         * if they are fundamental or vector-like.
         *
         * \tparam T_Type type of object
         */
        template<typename T_Type>
        struct GetInitializedInstance
        {
            typedef T_Type Type;

            template<typename ValueType>
            HDINLINE Type operator()(const ValueType& value) const
            {
                return Type(value);
            }
        };

    } // namespace traits
} // namespace pmacc
