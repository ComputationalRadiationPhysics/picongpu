/**
 * Copyright 2016 Heiko Burau
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "math/vector/Vector.hpp"
#include "pmacc_types.hpp"

namespace PMacc
{
namespace traits
{

/** Returns an initialized instance. If the type is a PMacc vector
 * all components are initialzed with the same value;
 *
 * \tparam T_Type type of object
 */
template<typename T_Type>
struct GetInitializedInstance
{
    typedef T_Type Type;

    HDINLINE Type operator()(const Type value) const
    {
        return Type(value);
    }
};

template<typename T_Type, int T_dim, typename T_Accessor, typename T_Navigator, template<typename, int> class T_Storage>
struct GetInitializedInstance<math::Vector<T_Type, T_dim, T_Accessor, T_Navigator, T_Storage> >
{
    typedef T_Type Type;
    typedef typename T_Type::type ValueType;

    HDINLINE Type operator()(const ValueType value) const
    {
        return Type::create(value);
    }
};

} // traits
} // PMacc
