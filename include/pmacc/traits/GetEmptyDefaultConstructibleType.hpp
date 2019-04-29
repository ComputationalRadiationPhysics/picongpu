/* Copyright 2015-2019 Rene Widera
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
namespace traits
{

/** Get type with empty default constructor
 *
 * the returned type must fulfill the points:
 *   1. empty constructor
 *   2. no default initialized member inside the empty constructor
 *   3. all member must fulfill point 1. and 2.
 *   4. all base classes must fulfill 1. and 2.
 *
 * The result is typical used to define structs/classes in the cuda shared memory
 *
 * @tparam T_Type any object (class or typename)
 *
 * @treturn ::type
 */
template<typename T_Type>
struct GetEmptyDefaultConstructibleType;


}//namespace traits

}//namespace pmacc
