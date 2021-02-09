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

#include <set>

#include "pmacc/types.hpp"

namespace pmacc
{
    class IEventData;
    class IEvent;

    /**
     * Implements an observable.
     */
    class EventNotify
    {
    public:
        virtual ~EventNotify()
        {
        }

        /**
         * Registers an observer at this observable.
         * @param event pointer to an observer implementing the IEvent interface.
         */
        void addObserver(IEvent* event)
        {
            observers.insert(event);
        }

        /**
         * Removes an observer from this observable.
         * @param event the observer to remove.
         */
        void removeObserver(IEvent* event)
        {
            observers.erase(event);
        }

        /**
         * Notifies all registered observers
         * @param eventId id of this notification
         * @param type the type of this notification
         * @param data data passed to observers
         */
        void notify(id_t eventId, EventType type, IEventData* data);

    private:
        std::set<IEvent*> observers;
    };

} // namespace pmacc
