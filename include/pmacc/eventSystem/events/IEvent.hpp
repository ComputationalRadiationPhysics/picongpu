/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

namespace pmacc
{
    class IEventData;

    /**
     * Interface for an observer.
     */
    class IEvent
    {
    public:
        /**
         * Destructor.
         */
        virtual ~IEvent()
        {
        }

        // IEventData *should* be small; using a pointer here will result in memory leaks...
        /**
         * Called when this observer is notified by the observable.
         * @param eventId id of the notification
         * @param type the type of the notification
         * @param data data passed from observable
         */
        virtual void event(id_t eventId, EventType type, IEventData* data) = 0;
    };

} // namespace pmacc
