/* Copyright 2015-2023 Rene Widera, Sergei Bastrakov
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

#include "pmacc/types.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pmacc
{
    namespace traits
    {
        /** Get next available type id
         *
         * Warning: is not thread-safe.
         */
        template<typename T_ResultType = uint64_t>
        static T_ResultType getUniqueId();

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
            const uint64_t TypeId<T_Type>::id = getUniqueId<uint64_t>();

            /** check if a value can be represented by the type
             *
             * @tparam T_ResultType type to cast the result to
             */
            template<typename T_ResultType>
            static void idRangeCheck(uint64_t id)
            {
                constexpr uint64_t maxValue = std::numeric_limits<T_ResultType>::max();

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
            }
        } // namespace detail

        /** Get next available type id
         *
         * @warning is not thread-safe.
         * @warning when using it to generate matching tags from multiple MPI ranks, ensure the same call order.
         *          For type-unique ids, GetUniqueTypeId<> is preferable as it is not dependent on the call order
         */
        template<typename T_ResultType>
        static T_ResultType getUniqueId()
        {
            uint64_t id = ++detail::counter();
            detail::idRangeCheck<T_ResultType>(id);
            return static_cast<T_ResultType>(id);
        }

        /** Get a unique id of a type
         *
         * - get a unique id of a type at runtime
         * - the id of a type is equal on each instance of a process
         *
         * @warning If you use different binaries compiled by different compilers within one MPI context the id will
         * not match between MPI ranks.
         *
         * @tparam T_Type any object (class or typename)
         * @tparam T_ResultType result type
         */
        template<typename T_Type, typename T_ResultType = uint64_t>
        struct GetUniqueTypeId
        {
            using ResultType = T_ResultType;
            using Type = T_Type;

            /** create unique id
             *
             * @param maxValue largest allowed id
             */
            static const ResultType uid(uint64_t maxValue = std::numeric_limits<ResultType>::max())
            {
                const uint64_t id = detail::TypeId<Type>::id;
                detail::idRangeCheck<T_ResultType>(id);
                return static_cast<ResultType>(id);
            }
        };

    } // namespace traits

} // namespace pmacc
