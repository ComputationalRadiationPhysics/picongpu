/**
 * Copyright 2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/types.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla_driver_types.hpp"

#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <chrono>

namespace cupla
{
namespace manager
{

namespace detail
{
    template<
        typename T_DeviceType,
        typename T_StreamType
    >
    class EmulatedEvent
    {
    private:
        bool hasTimer;

        using TimePoint = std::chrono::time_point<
            std::chrono::high_resolution_clock
        >;

        TimePoint time;

    public:
        using AlpakaEvent = ::alpaka::event::Event< T_StreamType >;
        std::unique_ptr< AlpakaEvent > event;

        EmulatedEvent( uint32_t flags ) :
            hasTimer( !( flags & cuplaEventDisableTiming ) ),
            event(
                new AlpakaEvent(
                    Device< T_DeviceType >::get().current()
                )
            )
        {

        }

        AlpakaEvent &
        operator *()
        {
            return *event;
        }

        void record( T_StreamType & stream )
        {
            ::alpaka::stream::enqueue( stream, *event );
            if( hasTimer )
            {
                ::alpaka::wait::wait( *event );
                time = std::chrono::high_resolution_clock::now();
            }
        }

        TimePoint getTimePoint() const
        {
            return time;
        }

        double elapsedSince( EmulatedEvent const & startEvent )
        {
            if( !hasTimer )
                std::cerr<<"event has no timing enabled"<<std::endl;

            std::chrono::duration<double, std::milli> timeElapsed_ms = time - startEvent.getTimePoint();
            return timeElapsed_ms.count();
        }

    };
}
    template<
        typename T_DeviceType,
        typename T_StreamType
    >
    struct Event
    {
        using DeviceType = T_DeviceType;
        using StreamType = T_StreamType;

        using EventType = detail::EmulatedEvent<
            DeviceType,
            StreamType
        >;

        using EventMap = std::map<
            cuplaEvent_t,
            std::unique_ptr<
                EventType
            >
        >;

        using MapVector = std::vector< EventMap >;

        MapVector m_mapVector;

        static auto
        get()
        -> Event &
        {
            static Event event;
            return event;
        }

        auto
        create( uint32_t flags )
        -> cuplaEvent_t
        {

            auto& device = Device< DeviceType >::get();

            std::unique_ptr<
                EventType
            > eventPtr(
                new EventType(
                    flags
                )
            );
            cuplaEvent_t eventId = reinterpret_cast< cuplaEvent_t >(
                m_mapVector[ device.id() ].size()
            );
            m_mapVector[ device.id() ].insert(
                std::make_pair( eventId, std::move( eventPtr ) )
            );
            return eventId;
        }

        auto
        event( cuplaEvent_t eventId = 0 )
        -> EventType &
        {
            auto& device = Device< DeviceType >::get();
            const auto deviceId = device.id();
            auto iter = m_mapVector[ deviceId ].find(
                eventId
            );

            if( iter == m_mapVector[ device.id( ) ].end() )
            {
                std::cerr << "event " << eventId <<
                    " not exists on device "<< deviceId << std::endl;
            }
            // @todo: check if stream was created
            return *(iter->second);
        }

        auto
        destroy( cuplaEvent_t eventId )
        -> bool
        {
            auto& device = Device< DeviceType >::get();
            const auto deviceId = device.id();

            auto iter = m_mapVector[ deviceId ].find(
                eventId
            );

            if( iter == m_mapVector[ deviceId ].end() )
            {
                std::cerr << "stream " << eventId <<
                    " can not destroyed (was never created) on device " <<
                    deviceId <<
                    std::endl;
                return false;
            }
            else
            {
                m_mapVector[ deviceId ].erase( iter );
                return true;
            }
        }

        /** delete all events on the current device
         *
         * @return true in success case else false
         */
        bool
        reset( )
        {
            auto& device = Device< DeviceType >::get();
            const auto deviceId = device.id();

            m_mapVector[ deviceId ].clear( );

            // @todo: check if clear creates errors
            return true;
        }


    protected:
        Event() :  m_mapVector( Device< DeviceType >::get().count() )
        {
        }

    };

} //namespace manager
} //namespace cupla
