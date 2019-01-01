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

#include "pmacc/types.hpp"
#include <sstream>
#include <string>
#include <stdexcept>
#include <boost/numeric/conversion/bounds.hpp>

namespace pmacc
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
    static uint64_t counter;
    static const uint64_t id;
};

template<>
const uint64_t GetUniqueTypeId<uint8_t>::id = 0;
template<>
uint64_t GetUniqueTypeId<uint8_t>::counter = 0;

template<typename T_Type>
const uint64_t GetUniqueTypeId<T_Type>::id = ++GetUniqueTypeId<uint8_t>::counter;

template<typename T_Type>
uint64_t GetUniqueTypeId<T_Type>::counter = GetUniqueTypeId<T_Type>::id;

} //namespace detail

/** Get a unique id of a type
 *
 * - generate a unique id of a type at *runtime*
 * - the id of a type is equal on each instance of a process
 *
 * @tparam T_Type any object (class or typename)
 * @tparam T_ResultType result type
 */
template<typename T_Type, typename T_ResultType = uint64_t>
struct GetUniqueTypeId
{
    typedef T_ResultType ResultType;
    typedef T_Type Type;

    /** create unique id
     *
     * @param maxValue largest allowed id
     */
    static const ResultType uid(uint64_t maxValue = boost::numeric::bounds<ResultType>::highest())
    {

        const uint64_t id = detail::GetUniqueTypeId<Type>::id;

        /* if `id` is out of range than throw an error */
        if (id > maxValue)
        {
            std::stringstream sId;
            sId << id;
            std::stringstream sMax;
            sMax << maxValue;
            throw std::runtime_error("generated id is out of range [ id = " +
                                     sId.str() +
                                     std::string(", largest allowed  id = ") +
                                     sMax.str() +
                                     std::string(" ]"));
        }
        return static_cast<ResultType> (id);
    }

};

}//namespace traits

}//namespace pmacc
