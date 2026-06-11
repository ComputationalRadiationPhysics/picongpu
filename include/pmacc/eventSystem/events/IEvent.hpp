/*
 * SPDX-FileCopyrightText: Felix Schmitt, Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
        virtual ~IEvent() = default;

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
