/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/eventSystem/events/IEvent.hpp"
#include "pmacc/eventSystem/events/IEventData.hpp"
#include "pmacc/types.hpp"

#include <set>

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
        virtual ~EventNotify() = default;

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
