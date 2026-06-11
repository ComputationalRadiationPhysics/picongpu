/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/EventTask.hpp"

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
         * Implementations of @see AsyncCommunication should specialize this,
         * but it is not intended to be called directly. Use @see AsyncCommunication
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
         * generic @see AsyncCommunicationImpl
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
