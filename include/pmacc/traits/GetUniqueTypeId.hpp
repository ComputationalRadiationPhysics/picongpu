/* Copyright 2015-2021 Rene Widera, Sergei Bastrakov
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
        /** Get next available type id
         *
         * Warning: is not thread-safe.
         */
        inline uint64_t getNextId();

        namespace detail
        {
            /** Global counter for type ids
             */
            inline uint64_t& counter()
            {
                static uint64_t value = 0;
                return value;
            }

            /** Unique id for a given type
             *
             * @tparam T_Type type
             */
            template<typename T_Type>
            struct TypeId
            {
                static const uint64_t id;
            };

            /** These id values are generated during the startup for all types that cause
             *  instantiation of GetUniqueTypeId<T_Type>::uid().
             *
             * The order of calls to GetUniqueTypeId<T_Type>::uid() does not affect the id
             * generation, which guarantees the ids are matching for all processes even when
             * the run-time access is not.
             */
            template<typename T_Type>
            const uint64_t TypeId<T_Type>::id = getNextId();

        } // namespace detail

        /** Get next available type id
         *
         * \warning is not thread-safe.
         * \warning when using it to generate matching tags from multiple MPI ranks, ensure the same call order.
         *          For type-unique ids, GetUniqueTypeId<> is preferable as it is not dependent on the call order
         */
        uint64_t getNextId()
        {
            return ++detail::counter();
        }

        /** Get a unique id of a type
         *
         * - get a unique id of a type at runtime
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
                const uint64_t id = detail::TypeId<Type>::id;

                /* if `id` is out of range than throw an error */
                if(id > maxValue)
                {
                    std::stringstream sId;
                    sId << id;
                    std::stringstream sMax;
                    sMax << maxValue;
                    throw std::runtime_error(
                        "generated id is out of range [ id = " + sId.str() + std::string(", largest allowed  id = ")
                        + sMax.str() + std::string(" ]"));
                }
                return static_cast<ResultType>(id);
            }
        };

    } // namespace traits

} // namespace pmacc
