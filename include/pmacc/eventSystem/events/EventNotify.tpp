/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz
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

#include "pmacc/eventSystem/events/EventNotify.hpp"
#include "pmacc/eventSystem/events/IEventData.hpp"
#include "pmacc/eventSystem/events/IEvent.hpp"
#include "pmacc/types.hpp"

#include <set>

namespace pmacc
{
    inline void EventNotify::notify(id_t eventId, EventType type, IEventData* data)
    {
        std::set<IEvent*>::iterator iter = observers.begin();
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
