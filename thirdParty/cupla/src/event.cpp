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


#include "cupla/namespace.hpp"
#include "cupla_runtime.hpp"
#include "cupla/manager/Memory.hpp"
#include "cupla/manager/Device.hpp"
#include "cupla/manager/Stream.hpp"
#include "cupla/manager/Event.hpp"
#include "cupla/api/event.hpp"

inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaEventCreateWithFlags(
    cuplaEvent_t * event,
    unsigned int flags
)
{
    *event = cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().create( flags );

    return cuplaSuccess;
}


CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaEventCreate(
    cuplaEvent_t * event
)
{
    *event = cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().create( 0 );

    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaEventDestroy( cuplaEvent_t event )
{
    if(
        cupla::manager::Event<
            cupla::AccDev,
            cupla::AccStream
        >::get().destroy( event )
    )
        return cuplaSuccess;
    else
        return cuplaErrorInitializationError;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaEventRecord(
    cuplaEvent_t event,
    cuplaStream_t stream
)
{
    auto& streamObject = cupla::manager::Stream<
        cupla::AccDev,
        cupla::AccStream
    >::get().stream( stream );
    auto& eventObject = cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().event( event );

    eventObject.record( streamObject );
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaEventElapsedTime(
    float * ms,
    cuplaEvent_t start,
    cuplaEvent_t end
)
{
    auto& eventStart = cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().event( start );
    auto& eventEnd = cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().event( end );
    *ms = static_cast< float >( eventEnd.elapsedSince( eventStart ) );
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaEventSynchronize(
    cuplaEvent_t event
)
{
    auto& eventObject = cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().event( event );
    ::alpaka::wait( *eventObject );
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaEventQuery( cuplaEvent_t event )
{
    auto& eventObject = cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().event( event );

    if( ::alpaka::isComplete( *eventObject ) )
    {
        return cuplaSuccess;
    }
    else
    {
        return cuplaErrorNotReady;
    }
}

} //namespace CUPLA_ACCELERATOR_NAMESPACE
