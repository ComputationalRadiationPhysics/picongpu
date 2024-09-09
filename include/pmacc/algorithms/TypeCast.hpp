/* Copyright 2013-2023 Heiko Burau, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"

namespace pmacc
{
    namespace algorithms
    {
        namespace precisionCast
        {
            template<typename CastToType, typename Type>
            struct TypeCast
            {
                using result = CastToType;

                constexpr result operator()(const Type& value) const
                {
                    return static_cast<result>(value);
                }
            };


            template<typename CastToType, typename Type>
            constexpr typename TypeCast<CastToType, Type>::result precisionCast(const Type& value)
            {
                return TypeCast<CastToType, Type>()(value);
            }

        } // namespace precisionCast
    } // namespace algorithms
} // namespace pmacc
