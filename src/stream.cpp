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

#include "cupla/api/stream.hpp"

inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaStreamCreate(
    cuplaStream_t * stream
)
{
    *stream = cupla::manager::Stream<
        cupla::AccDev,
        cupla::AccStream
    >::get().create();

    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaStreamDestroy( cuplaStream_t stream )
{
    if(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().destroy( stream )
    )
        return cuplaSuccess;
    else
        return cuplaErrorInitializationError;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaStreamSynchronize(
    cuplaStream_t stream
)
{
    auto& streamObject = cupla::manager::Stream<
        cupla::AccDev,
        cupla::AccStream
    >::get().stream( stream );
    ::alpaka::wait::wait( streamObject );
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaStreamWaitEvent(
    cuplaStream_t stream,
    cuplaEvent_t event,
    unsigned int
)
{
    auto& streamObject = cupla::manager::Stream<
        cupla::AccDev,
        cupla::AccStream
    >::get().stream( stream );

    auto& eventObject = *cupla::manager::Event<
        cupla::AccDev,
        cupla::AccStream
    >::get().event( event );

    ::alpaka::wait::wait(streamObject,eventObject);
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaStreamQuery( cuplaStream_t stream )
{
    auto& streamObject = cupla::manager::Stream<
        cupla::AccDev,
        cupla::AccStream
    >::get().stream( stream );

    if( alpaka::queue::empty( streamObject ) )
        return cuplaSuccess;
    else
        return cuplaErrorNotReady;
}

} //namespace CUPLA_ACCELERATOR_NAMESPACE
