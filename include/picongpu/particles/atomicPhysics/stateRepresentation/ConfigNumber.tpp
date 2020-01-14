/* Copyright 2020 Brian Marre
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

/** this file specfies how an object of the ConfigNumber class can be written to
 * an external file for storage.
*/

namespace pmacc
{
namespace  traits
{

//defines what datatype is to be used to save data in this object
template< typename T_DataType, uint8_t T_NumberLevels >
struct GetComponents< picongpu::particles::atomicPhysics::stateRepresentation<
        T_DataType,
        T_NumberLevels
        >,
    false
    >
{
    using type = typename T_DataType;
}

//defines how many independent components are saved in the object
template< typename T_DataType, uint8_t T_NumberLevels >
struct GetNComponents< picongpu::particles::atomicPhysics::stateRepresentation<
        T_DataType,
        T_NumberLevels
        >,
    false
    >
{
    static constexpr uint32_t value = 1u;
}

}//namespace traits
}//namespace pmacc