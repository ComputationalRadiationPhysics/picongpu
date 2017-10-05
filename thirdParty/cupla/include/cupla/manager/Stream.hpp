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

#include <map>
#include <vector>
#include <memory>

namespace cupla
{
namespace manager
{

    template<
        typename T_DeviceType,
        typename T_StreamType
    >
    struct Stream
    {
        using DeviceType = T_DeviceType;
        using StreamType = T_StreamType;


        using StreamMap = std::map<
            cuplaStream_t,
            std::unique_ptr<
                StreamType
            >
        >;
        using MapVector = std::vector< StreamMap >;

        MapVector m_mapVector;

        static auto
        get()
        -> Stream &
        {
            static Stream stream;
            return stream;
        }

        auto
        create( )
        -> cuplaStream_t
        {

            auto& device = Device< DeviceType >::get();

            std::unique_ptr<
                StreamType
            > streamPtr(
                new StreamType(
                    device.current()
                )
            );
            cuplaStream_t streamId = reinterpret_cast< cuplaStream_t >(
                m_mapVector[ device.id() ].size()
            );
            m_mapVector[ device.id() ].insert(
                std::make_pair( streamId, std::move( streamPtr ) )
            );
            return streamId;
        }

        auto
        stream( cuplaStream_t streamId = 0 )
        -> StreamType &
        {
            auto& device = Device< DeviceType >::get();
            const auto deviceId = device.id();
            auto iter = m_mapVector[ deviceId ].find(
                streamId
            );

            if( iter == m_mapVector[ device.id( ) ].end() )
            {
                if( streamId == 0 )
                {
                    this->create( );
                    return this->stream( streamId );
                }
                else
                {
                    std::cerr << "stream " << streamId <<
                        " not exists on device "<< deviceId << std::endl;
                }
            }
            // @todo: check if stream was created
            return *(iter->second);
        }

        auto
        destroy( cuplaStream_t streamId)
        -> bool
        {
            auto& device = Device< DeviceType >::get();
            const auto deviceId = device.id();

            auto iter = m_mapVector[ deviceId ].find(
                streamId
            );

            if( iter == m_mapVector[ deviceId ].end() )
            {
                std::cerr << "stream " << streamId <<
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


        /** delete all streams on the current device
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
        Stream() :  m_mapVector( Device< DeviceType >::get().count() )
        {
        }

    };

} //namespace manager
} //namespace cupla
