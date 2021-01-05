/* Copyright 2015-2021 Alexander Grund
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
    namespace communication
    {
        /**
         * Wrapper to convert a bool into a type
         */
        template<bool T_value>
        struct Bool2Type;

        /**
         * Implementations of \see AsyncCommunication should specialize this,
         * but it is not intended to be called directly. Use \see AsyncCommunication
         *
         * The 2nd template parameter can be used to check for conditions on
         * templated implementations. E.g.:
         *
         *     template<typename T_Data>
         *     struct AsyncCommunicationImpl<
         *         T_Data,
         *         Bool2Type< boost::is_integral<T_Data>::value >
         *     >{...}
         */
        template<typename T_Data, typename T_IsSpecialized = Bool2Type<true>>
        struct AsyncCommunicationImpl;

        /**
         * This policy starts an asynchronous communication of the given data
         * (e.g. a particle species)
         *
         * It must be a functor with signature EventTask(T_Data&, EventTask parentEvent)
         * but can be templated (again) over T_Data to get the actual type. This
         * is helpful for generic implementations that apply to T_Data and all
         * derived classes but want to use the possibly more derived type
         *
         * For different T_Data types you can either specialize this or the more
         * generic \see AsyncCommunicationImpl
         */
        template<typename T_Data>
        struct AsyncCommunication : public AsyncCommunicationImpl<T_Data>
        {
        };

        template<typename T_Data>
        EventTask asyncCommunication(T_Data& data, EventTask parent)
        {
            return AsyncCommunication<T_Data>()(data, parent);
        }

    } // namespace communication
} // namespace pmacc
