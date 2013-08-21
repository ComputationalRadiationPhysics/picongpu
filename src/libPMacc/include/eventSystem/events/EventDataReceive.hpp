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
 

#ifndef _EVENTDATARECEIVE_HPP
#define	_EVENTDATARECEIVE_HPP

#include "eventSystem/events/IEventData.hpp"

namespace PMacc
{

    class EventDataReceive : public IEventData
    {
    public:

        EventDataReceive(EventNotify *task, size_t recv_count) :
        IEventData(task),
        recv_count(recv_count)
        {

        }

        size_t getReceivedCount() const
        {
            return recv_count;
        }

    private:
        size_t recv_count;

    };

} //namespace PMacc

#endif	/* _EVENTDATARECEIVE_HPP */

