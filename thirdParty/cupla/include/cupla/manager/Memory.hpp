/* Copyright 2016 Rene Widera
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

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"
#include "cupla/manager/Device.hpp"

#include <vector>
#include <map>
#include <memory>
#include <utility>

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
namespace manager
{

    template<
        typename T_DeviceType,
        typename T_Dim
    >
    struct Memory
    {
        using DeviceType = T_DeviceType;
        static constexpr uint32_t dim = T_Dim::value;

        using BufType = ::alpaka::Buf<
            DeviceType,
            uint8_t,
            T_Dim,
            MemSizeType
        >;

        using MemoryMap = std::map<
            uint8_t*,
            std::unique_ptr<
                BufType
            >
        >;

        using MapVector = std::vector< MemoryMap >;

        MapVector m_mapVector;

        static auto
        get()
        -> Memory &
        {
            static Memory mem;
            return mem;
        }


        auto
        alloc(
            MemVec< dim > const & extent
        )
        -> BufType &
        {

            auto& device = Device< DeviceType >::get();

            std::unique_ptr<
                BufType
            > bufPtr(
                new BufType(
                    ::alpaka::allocBuf<uint8_t, MemSizeType>(
                         device.current(),
                         extent
                    )
                )
            );


            uint8_t *nativePtr = ::alpaka::getPtrNative(*bufPtr);
            m_mapVector[ device.id() ].insert(
                std::make_pair( nativePtr, std::move( bufPtr ) )
            );
            return *m_mapVector[ device.id() ][ nativePtr ];
        }

        auto
        free( void * ptr)
        -> bool
        {
            if( ptr == nullptr)
                return true;

            auto& device = Device< DeviceType >::get();
            const auto deviceId = device.id();

            auto iter = m_mapVector[ deviceId ].find(
                static_cast< uint8_t * >( ptr )
            );

            if( iter == m_mapVector[ deviceId ].end() )
            {
                return false;
            }
            else
            {
                m_mapVector[ deviceId ].erase( iter );
                return true;
            }
        }

        /** delete all memory on the current device
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
        Memory() : m_mapVector( Device< DeviceType >::get().count() )
        {

        }

    };

} //namespace manager
} //namespace CUPLA_ACCELERATOR_NAMESPACE
} //namespace cupla
