/**
 * Copyright 2013-2015 Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "eventSystem/events/EventNotify.hpp"
#include "eventSystem/events/IEventData.hpp"
#include "eventSystem/events/IEvent.hpp"
#include "types.h"

#include <set>

namespace PMacc
{

        inline void EventNotify::notify( id_t eventId, EventType type, IEventData *data )
        {
            std::set<IEvent*>::iterator iter = observers.begin( );
            for (; iter != observers.end( ); iter++ )
            {
                if ( *iter != NULL )
                    ( *iter )->event( eventId, type, data );
            }
            /* if notify is not called from destructor
             * other tasks can register after this call.
             * But any ITask must call this function in destrctor again"
             */
            observers.clear( );

            /**
             * \TODO are we sure that data won't be deleted anywhere else?
             * if (data != NULL)
             *  delete data;
             **/

        }

} //namespace PMacc
