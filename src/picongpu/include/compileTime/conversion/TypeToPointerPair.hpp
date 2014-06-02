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

#include "types.h"

#include <boost/mpl/pair.hpp>


namespace picongpu
{

/** Wrapper to use any type as identifier
 *
 * Wrapp a type without
 */
template<typename T_Type>
struct TypeAsIdentifier
{
    typedef T_Type type;
};

template<typename T_Type>
struct Cover
{
    typedef TypeAsIdentifier<T_Type> type;
};

/** create boost mpl pair
 *
 * @tparam T_Type any type
 * @resturn ::type boost mpl pair where first s set to T_Type
 */
template<typename T_Type>
struct TypeToPointerPair
{
    typedef T_Type* TypePtr;
    typedef bmpl::pair< typename Cover<T_Type>::type , TypePtr > type;
};

}//namespace picongpu
