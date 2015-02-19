/**
 * Copyright 2013 Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


namespace picongpu
{

namespace traits
{
/** Get the value of a unit
 *
 * \tparam T_Identifier any identifier
 */
template<typename T_Identifier>
struct GetUnitValue;

} //namespace traits

template<typename T_Identifier>
HDINLINE float_64 getUnitValue()
{
    return traits::GetUnitValue<T_Identifier>()();
}

template<typename T_Identifier>
HDINLINE float_64 getUnitValue(const T_Identifier&)
{
    return getUnitValue<T_Identifier>();
}

}// namespace picongpu
