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


#include "cupla_runtime.hpp"
#include "cupla/manager/Memory.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla/manager/Stream.hpp"
#include "cupla/manager/Event.hpp"
#include "cupla/api/device.hpp"
#include <stdexcept>

cuplaError_t
cuplaGetDeviceCount( int * count)
{
    *count = cupla::manager::Device< cupla::AccDev >::get().count();
    return cuplaSuccess;
}

cuplaError_t
cuplaSetDevice( int idx)
{
    try
    {
      cupla::manager::Device< cupla::AccDev >::get().device( idx );
    }
    catch(const std::system_error& e)
    {
      return static_cast<cuplaError_t>( e.code().value() );
    }
    return cuplaSuccess;
}

cuplaError_t
cuplaGetDevice( int * deviceId )
{
    *deviceId = cupla::manager::Device< cupla::AccDev >::get().id();
    return cuplaSuccess;
}

cuplaError_t
cuplaDeviceReset( )
{
    // wait that all work on the device is finished
    cuplaDeviceSynchronize( );

    // delete all events on the current device
    cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().reset( );

    // delete all memory on the current device
    cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim<1u>
    >::get().reset( );

    cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim<2u>
    >::get().reset( );

    cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim<3u>
    >::get().reset( );

    // delete all streams on the current device
    cupla::manager::Stream<
        cupla::AccDev,
        cupla::AccStream
    >::get().reset( );

    cupla::manager::Device< cupla::AccDev >::get( ).reset( );
    return cuplaSuccess;
}

cuplaError_t
cuplaDeviceSynchronize( )
{
    ::alpaka::wait::wait(
        cupla::manager::Device< cupla::AccDev >::get( ).current( )
    );
    return cuplaSuccess;
}

cuplaError_t
cuplaMemGetInfo(
    size_t * free,
    size_t * total
)
{
    auto& device(
        cupla::manager::Device<
            cupla::AccDev
        >::get().current()
    );
    *total = ::alpaka::dev::getMemBytes( device );
    *free = ::alpaka::dev::getFreeMemBytes( device );
    return cuplaSuccess;
}
