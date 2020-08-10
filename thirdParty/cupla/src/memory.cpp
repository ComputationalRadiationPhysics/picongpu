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
#include "cupla/api/memory.hpp"

inline namespace CUPLA_ACCELERATOR_NAMESPACE
{

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMalloc(
    void **ptrptr,
    size_t size
)
{

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<1u>,
        cupla::MemSizeType
    > extent( size );

    auto& buf = cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim<1u>
    >::get().alloc( extent );

    // @toto catch errors
    *ptrptr = ::alpaka::mem::view::getPtrNative(buf);
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMallocPitch(
    void ** devPtr,
    size_t * pitch,
    size_t const width,
    size_t const height
)
{
    const ::alpaka::vec::Vec<
        cupla::AlpakaDim< 2u >,
        cupla::MemSizeType
    > extent( height, width );

    auto& buf = cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim< 2u >
    >::get().alloc( extent );

    // @toto catch errors
    *devPtr = ::alpaka::mem::view::getPtrNative(buf);
    *pitch = ::alpaka::mem::view::getPitchBytes< 1u >( buf );

    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMalloc3D(
    cuplaPitchedPtr * const pitchedDevPtr,
    cuplaExtent const extent
)
{

    auto& buf = cupla::manager::Memory<
        cupla::AccDev,
        cupla::AlpakaDim< 3u >
    >::get().alloc( extent );

    // @toto catch errors
    *pitchedDevPtr = make_cuplaPitchedPtr(
        ::alpaka::mem::view::getPtrNative(buf),
        ::alpaka::mem::view::getPitchBytes< 2u >( buf ),
        extent.width,
        extent.height
    );

    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaExtent
make_cuplaExtent(
    size_t const w,
    size_t const h,
    size_t const d
)
{
    return cuplaExtent( w, h, d );
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaPos
make_cuplaPos(
    size_t const x,
    size_t const y,
    size_t const z
)
{
    return cuplaPos( x, y, z );
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaPitchedPtr
make_cuplaPitchedPtr(
    void * const d,
    size_t const p,
    size_t const xsz,
    size_t const ysz
)
{
    return cuplaPitchedPtr( d, p, xsz, ysz );
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMallocHost(
    void **ptrptr,
    size_t size
)
{
    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<1u>,
        cupla::MemSizeType
    > extent( size );

    auto& buf = cupla::manager::Memory<
        cupla::AccHost,
        cupla::AlpakaDim<1u>
    >::get().alloc( extent );

    prepareForAsyncCopy( buf );

    // @toto catch errors
    *ptrptr = ::alpaka::mem::view::getPtrNative(buf);
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t cuplaFree(void *ptr)
{

    if(
        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<1u>
        >::get().free( ptr )
    )
        return cuplaSuccess;
    else if(
        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<2u>
        >::get().free( ptr )
    )
        return cuplaSuccess;
    else if(
        cupla::manager::Memory<
            cupla::AccDev,
            cupla::AlpakaDim<3u>
        >::get().free( ptr )
    )
        return cuplaSuccess;
    else
        return cuplaErrorMemoryAllocation;

}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t cuplaFreeHost(void *ptr)
{

    if(
        cupla::manager::Memory<
            cupla::AccHost,
            cupla::AlpakaDim<1u>
        >::get().free( ptr )
    )
        return cuplaSuccess;
    else
        return cuplaErrorMemoryAllocation;

}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t cuplaMemcpyAsync(
    void *dst,
    const void *src,
    size_t count,
    enum cuplaMemcpyKind kind,
    cuplaStream_t stream
)
{
    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<1u>,
        cupla::MemSizeType
    > numBytes(count);

    auto& device(
        cupla::manager::Device<
            cupla::AccDev
        >::get().current()
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( stream )
    );

    switch(kind)
    {
        case cuplaMemcpyHostToDevice:
        {
            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );

            const cupla::HostBufWrapper< 1u > hBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                host,
                numBytes
            );
            cupla::DeviceBufWrapper< 1u > dBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                device,
                numBytes
            );

            ::alpaka::mem::view::copy(
                streamObject,
                dBuf,
                hBuf,
                numBytes
            );

        }
            break;
        case cuplaMemcpyDeviceToHost:
        {
            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );
            const cupla::DeviceBufWrapper< 1u > dBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                device,
                numBytes
            );
            cupla::HostBufWrapper< 1u > hBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                host,
                numBytes
            );

            ::alpaka::mem::view::copy(
                streamObject,
                hBuf,
                dBuf,
                numBytes
            );

        }
            break;
        case cuplaMemcpyDeviceToDevice:
        {
            const cupla::DeviceBufWrapper< 1u > dSrcBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                device,
                numBytes
            );
            cupla::DeviceBufWrapper< 1u > dDestBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                device,
                numBytes
            );

            ::alpaka::mem::view::copy(
                streamObject,
                dDestBuf,
                dSrcBuf,
                numBytes
            );

        }
            break;
        case cuplaMemcpyHostToHost:
        {
            auto& hostStreamObject(
                cupla::manager::Stream<
                    cupla::AccHost,
                    cupla::AccHostStream
                >::get().stream( stream )
            );
            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );
            const cupla::HostBufWrapper< 1u > hSrcBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                host,
                numBytes
            );
            cupla::HostBufWrapper< 1u > hDestBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                host,
                numBytes
            );

            ::alpaka::mem::view::copy(
                hostStreamObject,
                hDestBuf,
                hSrcBuf,
                numBytes
            );

        }
        break;
    }
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMemcpy(
    void *dst,
    const void *src,
    size_t count,
    enum cuplaMemcpyKind kind
)
{
    cuplaDeviceSynchronize();

    cuplaMemcpyAsync(
        dst,
        src,
        count,
        kind,
        0
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( 0 )
    );
    ::alpaka::wait::wait( streamObject );

    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMemsetAsync(
    void * devPtr,
    int value,
    size_t count,
    cuplaStream_t stream
)
{
    auto& device(
        cupla::manager::Device<
            cupla::AccDev
        >::get().current()
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( stream )
    );

    ::alpaka::vec::Vec<
        cupla::AlpakaDim<1u>,
        cupla::MemSizeType
    > const
    numBytes(count);

    cupla::DeviceBufWrapper< 1u >
    dBuf(
        static_cast< uint8_t * >( devPtr ),
        device,
        numBytes
    );

    ::alpaka::mem::view::set(
        streamObject,
        dBuf,
        value,
        numBytes
    );

    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMemset(
    void * devPtr,
    int value,
    size_t count
)
{
    cuplaDeviceSynchronize();

    cuplaMemsetAsync(
        devPtr,
        value,
        count,
        0
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( 0 )
    );
    ::alpaka::wait::wait( streamObject );

    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMemcpy2DAsync(
    void * dst,
    size_t const dPitch,
    void const * const src,
    size_t const sPitch,
    size_t const width,
    size_t const height,
    enum cuplaMemcpyKind kind,
    cuplaStream_t const stream
)
{
    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<2u>,
        cupla::MemSizeType
    > numBytes( height, width );

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<2u>,
        cupla::MemSizeType
    > dstPitch( dPitch * height , dPitch );

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<2u>,
        cupla::MemSizeType
    > srcPitch( sPitch * height , sPitch );

    auto& device(
        cupla::manager::Device<
            cupla::AccDev
        >::get().current()
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( stream )
    );

    switch(kind)
    {
        case cuplaMemcpyHostToDevice:
        {
            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );

            const cupla::HostBufWrapper< 2u > hBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                host,
                numBytes,
                srcPitch
            );
            cupla::DeviceBufWrapper< 2u > dBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                device,
                numBytes,
                dstPitch
            );

            ::alpaka::mem::view::copy(
                streamObject,
                dBuf,
                hBuf,
                numBytes
            );

        }
            break;
        case cuplaMemcpyDeviceToHost:
        {
            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );
            const cupla::DeviceBufWrapper< 2u > dBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                device,
                numBytes,
                srcPitch
            );
            cupla::HostBufWrapper< 2u > hBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                host,
                numBytes,
                dstPitch
            );

            ::alpaka::mem::view::copy(
                streamObject,
                hBuf,
                dBuf,
                numBytes
            );

        }
            break;
        case cuplaMemcpyDeviceToDevice:
        {
            const cupla::DeviceBufWrapper< 2u > dSrcBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                device,
                numBytes,
                srcPitch
            );
            cupla::DeviceBufWrapper< 2u > dDestBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                device,
                numBytes,
                dstPitch
            );

            ::alpaka::mem::view::copy(
                streamObject,
                dDestBuf,
                dSrcBuf,
                numBytes
            );

        }
        break;
        case cuplaMemcpyHostToHost:
        {
             auto& hostStreamObject(
                cupla::manager::Stream<
                    cupla::AccHost,
                    cupla::AccHostStream
                >::get().stream( stream )
            );
            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );
            const cupla::HostBufWrapper< 2u > hSrcBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(src)
                ),
                host,
                numBytes,
                srcPitch
            );
            cupla::HostBufWrapper< 2u > hDestBuf(
                static_cast<uint8_t *>(
                    dst
                ),
                host,
                numBytes,
                dstPitch
            );

            ::alpaka::mem::view::copy(
                hostStreamObject,
                hDestBuf,
                hSrcBuf,
                numBytes
            );

        }
        break;
    }
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMemcpy2D(
    void * dst,
    size_t const dPitch,
    void const * const src,
    size_t const sPitch,
    size_t const width,
    size_t const height,
    enum cuplaMemcpyKind kind
)
{
    cuplaDeviceSynchronize();

    cuplaMemcpy2DAsync(
        dst,
        dPitch,
        src,
        sPitch,
        width,
        height,
        kind,
        0
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( 0 )
    );
    ::alpaka::wait::wait( streamObject );

    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMemcpy3DAsync(
    const cuplaMemcpy3DParms * const p,
    cuplaStream_t stream
)
{
    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<3u>,
        cupla::MemSizeType
    > numBytes( p->extent );

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<3u>,
        cupla::MemSizeType
    > extentSrc(
        p->srcPtr.xsize * p->srcPtr.ysize * ( p->extent.depth + p->srcPos.z ),
        p->srcPtr.xsize * p->srcPtr.ysize,
        p->srcPtr.xsize
    );

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<3u>,
        cupla::MemSizeType
    > extentDst(
        p->dstPtr.xsize * p->dstPtr.ysize * ( p->extent.depth + p->dstPos.z ),
        p->dstPtr.xsize * p->dstPtr.ysize,
        p->dstPtr.xsize
    );

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<3u>,
        cupla::MemSizeType
    > offsetSrc(
        p->srcPos.z,
        p->srcPos.y,
        p->srcPos.x
    );

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<3u>,
        cupla::MemSizeType
    > offsetDst(
        p->dstPos.z,
        p->dstPos.y,
        p->dstPos.x
    );

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<3u>,
        cupla::MemSizeType
    > dstPitch(
        p->dstPtr.pitch * p->dstPtr.ysize * ( p->extent.depth + p->dstPos.z ), // @todo: can't create z pitch,  but is not needed by alpaka
        p->dstPtr.pitch * p->dstPtr.ysize,
        p->dstPtr.pitch
    );

    const ::alpaka::vec::Vec<
        cupla::AlpakaDim<3u>,
        cupla::MemSizeType
    > srcPitch(
        p->srcPtr.pitch * p->srcPtr.ysize * ( p->extent.depth + p->srcPos.z ), // @todo: can't create z pitch, but is not needed by alpaka
        p->srcPtr.pitch * p->srcPtr.ysize,
        p->srcPtr.pitch
    );

    auto& device(
        cupla::manager::Device<
            cupla::AccDev
        >::get().current()
    );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( stream )
    );

    switch(p->kind)
    {
        case cuplaMemcpyHostToDevice:
        {
            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );

            cupla::HostBufWrapper< 3u > hBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(p->srcPtr.ptr)
                ),
                host,
                extentSrc,
                srcPitch
            );
            cupla::DeviceBufWrapper< 3u > dBuf(
                static_cast<uint8_t *>(
                    p->dstPtr.ptr
                ),
                device,
                extentDst,
                dstPitch
            );

            cupla::DeviceViewWrapper< 3u > dView(
                dBuf,
                extentDst - offsetDst,
                offsetDst
            );

            ::alpaka::mem::view::copy(
                streamObject,
                dView,
                cupla::HostViewWrapper< 3u >(
                    hBuf,
                    extentSrc - offsetSrc,
                    offsetSrc
                ),
                numBytes
            );

        }
            break;
        case cuplaMemcpyDeviceToHost:
        {
            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );
            cupla::DeviceBufWrapper< 3u > dBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(p->srcPtr.ptr)
                ),
                device,
                extentSrc,
                srcPitch
            );
            cupla::HostBufWrapper< 3u > hBuf(
                static_cast<uint8_t *>(
                    p->dstPtr.ptr
                ),
                host,
                extentDst,
                dstPitch
            );

            cupla::HostViewWrapper< 3u > hView(
                hBuf,
                extentDst - offsetDst,
                offsetDst
            );

            ::alpaka::mem::view::copy(
                streamObject,
                hView,
                cupla::DeviceViewWrapper< 3u >(
                    dBuf,
                    extentSrc - offsetSrc,
                    offsetSrc
                ),
                numBytes
            );

        }
            break;
        case cuplaMemcpyDeviceToDevice:
        {
            cupla::DeviceBufWrapper< 3u > dSrcBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(p->srcPtr.ptr)
                ),
                device,
                extentSrc,
                srcPitch
            );
            cupla::DeviceBufWrapper< 3u > dDestBuf(
                static_cast<uint8_t *>(
                    p->dstPtr.ptr
                ),
                device,
                extentDst,
                dstPitch
            );

            cupla::DeviceViewWrapper< 3u > dView(
                dDestBuf,
                extentDst - offsetDst,
                offsetDst
            );

            ::alpaka::mem::view::copy(
                streamObject,
                dView,
                cupla::DeviceViewWrapper< 3u >(
                    dSrcBuf,
                    extentSrc - offsetSrc,
                    offsetSrc
                ),
                numBytes
            );

        }
        break;
        case cuplaMemcpyHostToHost:
        {
            auto& hostStreamObject(
                cupla::manager::Stream<
                    cupla::AccHost,
                    cupla::AccHostStream
                >::get().stream( stream )
            );

            auto& host(
                cupla::manager::Device<
                    cupla::AccHost
                >::get().current()
            );
            cupla::HostBufWrapper< 3u > hSrcBuf(
                const_cast<uint8_t *>(
                    static_cast<const uint8_t *>(p->srcPtr.ptr)
                ),
                host,
                extentSrc,
                srcPitch
            );
            cupla::HostBufWrapper< 3u > hDestBuf(
                static_cast<uint8_t *>(
                    p->dstPtr.ptr
                ),
                host,
                extentDst,
                dstPitch
            );

            cupla::HostViewWrapper< 3u > hView(
                hDestBuf,
                extentDst - offsetDst,
                offsetDst
            );
            ::alpaka::mem::view::copy(
                hostStreamObject,
                hView,
                cupla::HostViewWrapper< 3u >(
                    hSrcBuf,
                    extentSrc - offsetSrc,
                    offsetSrc
                ),
                numBytes
            );

        }
        break;
    }
    return cuplaSuccess;
}

CUPLA_HEADER_ONLY_FUNC_SPEC
cuplaError_t
cuplaMemcpy3D(
    const cuplaMemcpy3DParms * const p
)
{
    cuplaDeviceSynchronize();

    cuplaMemcpy3DAsync( p, 0 );

    auto& streamObject(
        cupla::manager::Stream<
            cupla::AccDev,
            cupla::AccStream
        >::get().stream( 0 )
    );
    ::alpaka::wait::wait( streamObject );

    return cuplaSuccess;
}

} //namespace CUPLA_ACCELERATOR_NAMESPACE
