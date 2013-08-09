/**
 * Copyright 2013 Felix Schmitt, René Widera, Wolfgang Hoenig
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 
/* 
 * File:   EventNotify.hpp
 * Author: whoenig
 *
 * Created on 9. April 2010, 10:17
 */

#ifndef _EVENTNOTIFY_HPP
#define	_EVENTNOTIFY_HPP

#include <set>
#include "types.h"
#include "eventSystem/events/IEvent.hpp"



namespace PMacc
{

    class IEventData;
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
        void notify(id_t eventId, EventType type, IEventData *data);

    private:
        std::set<IEvent*> observers;

    };

} //namespace PMacc


#endif	/* _EVENTNOTIFY_HPP */

