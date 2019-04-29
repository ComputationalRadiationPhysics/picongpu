/* Copyright 2014-2019 Rene Widera
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

/** Get resolved type
 *
 * Explicitly resolve the type of a synonym type, e.g., resolve the type of an PMacc alias.
 * A synonym type is wrapper type (class) around an other type.
 * If this trait is not defined for the given type the result is the identity of the given type.
 *
 * @tparam T_Object any object (class or typename)
 *
 * @treturn ::type
 */
template<typename T_Object>
struct Resolve
{
    typedef T_Object type;
};

}//namespace traits

}//namespace pmacc
