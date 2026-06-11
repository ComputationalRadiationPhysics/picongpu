/*
 * SPDX-FileCopyrightText: Rene Widera, Benjamin Worpitz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "pmacc/eventSystem/events/EventNotify.hpp"

namespace pmacc
{
    void EventNotify::notify(id_t eventId, EventType type, IEventData* data)
    {
        auto iter = observers.begin();
        for(; iter != observers.end(); iter++)
        {
            if(*iter != nullptr)
                (*iter)->event(eventId, type, data);
        }
        /* if notify is not called from destructor
         * other tasks can register after this call.
         * But any ITask must call this function in destrctor again"
         */
        observers.clear();

        /**
         * \TODO are we sure that data won't be deleted anywhere else?
         * if (data != nullptr)
         *  delete data;
         **/
    }

} // namespace pmacc
