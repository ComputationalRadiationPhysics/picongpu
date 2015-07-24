/**
 * Copyright 2015 Rene Widera
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

#include "types.h"

namespace PMacc
{
namespace traits
{

namespace detail
{

/** create unique id
 *
 * id is not beginning with zero
 *
 * This class is based on
 *  - http://stackoverflow.com/a/7562583
 *  - author: MSN (edited version from Sep 27th 2011 at 3:59)
 *  - license: CC-BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0/)
 */
template<typename T_Type>
struct GetUniqueTypeId
{
    static uint8_t byte;
    static const uint64_t id;
};
template<typename T_Type>
uint8_t GetUniqueTypeId<T_Type>::byte;

template<typename T_Type>
const uint64_t GetUniqueTypeId<T_Type>::id = uint64_t(&GetUniqueTypeId<T_Type>::byte);
} //namespace detail

/** Get a unique id of a type
 *
 * - generate a unique 64bit id of a type at *runtime*
 * - the id of a type is equal on each instance of a process
 *
 * @tparam T_Type any object (class or typename)
 *
 * @treturn ::uid
 */
template<typename T_Type>
struct GetUniqueTypeId
{
    static const uint64_t uid;
};

/** instantiation of traits::GetUniqueTypeId
 *
 * - create a instance of `traits::GetUniqueTypeId` and initialize the uid
 * - map `detail::GetUniqueTypeId<T>` to a range [1; 2^64-1]
 *   `uid = unique_id_of_current_type - base_unique_id`
 */
template<typename T_Type>
const uint64_t GetUniqueTypeId<T_Type>::uid = detail::GetUniqueTypeId<T_Type>::id - detail::GetUniqueTypeId<uint8_t>::id;

}//namespace traits

}//namespace PMacc
