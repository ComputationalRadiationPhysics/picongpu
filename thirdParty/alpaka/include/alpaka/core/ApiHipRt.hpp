/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <boost/predef.h>

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <hip/hip_runtime_api.h>
#    include <hip/hip_version.h>

namespace alpaka
{
    struct ApiHipRt
    {
        // Names
        static constexpr char name[] = "Hip";
        static constexpr auto version = BOOST_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0);

        // Types
        using DeviceAttr_t = ::hipDeviceAttribute_t;
        using DeviceProp_t = ::hipDeviceProp_t;
        using Error_t = ::hipError_t;
        using Event_t = ::hipEvent_t;
        using Extent_t = ::hipExtent;
        using Flag_t = unsigned int;
        using FuncAttributes_t = ::hipFuncAttributes;
        using HostFn_t = void (*)(void* data); // same as hipHostFn_t
        using Limit_t = ::hipLimit_t;
        using Memcpy3DParms_t = ::hipMemcpy3DParms;
        using MemcpyKind_t = ::hipMemcpyKind;
        using PitchedPtr_t = ::hipPitchedPtr;
        using Pos_t = ::hipPos;
        using Stream_t = ::hipStream_t;

        // Constants
        static constexpr Error_t success = ::hipSuccess;
        static constexpr Error_t errorNotReady = ::hipErrorNotReady;
        static constexpr Error_t errorHostMemoryAlreadyRegistered = ::hipErrorHostMemoryAlreadyRegistered;
        static constexpr Error_t errorHostMemoryNotRegistered = ::hipErrorHostMemoryNotRegistered;
        static constexpr Error_t errorUnsupportedLimit = ::hipErrorUnsupportedLimit;
        static constexpr Error_t errorUnknown = ::hipErrorUnknown;

        static constexpr Flag_t eventDefault = hipEventDefault;
        static constexpr Flag_t eventBlockingSync = hipEventBlockingSync;
        static constexpr Flag_t eventDisableTiming = hipEventDisableTiming;
        static constexpr Flag_t eventInterprocess = hipEventInterprocess;

        static constexpr Flag_t hostMallocDefault = hipHostMallocDefault;
        static constexpr Flag_t hostMallocMapped = hipHostMallocMapped;
        static constexpr Flag_t hostMallocPortable = hipHostMallocPortable;
        static constexpr Flag_t hostMallocWriteCombined = hipHostMallocWriteCombined;
        static constexpr Flag_t hostMallocCoherent = hipHostMallocCoherent;
        static constexpr Flag_t hostMallocNonCoherent = hipHostMallocNonCoherent;

        static constexpr Flag_t hostRegisterDefault = hipHostRegisterDefault;
        static constexpr Flag_t hostRegisterPortable = hipHostRegisterPortable;
        static constexpr Flag_t hostRegisterMapped = hipHostRegisterMapped;
        static constexpr Flag_t hostRegisterIoMemory = hipHostRegisterIoMemory;

        static constexpr MemcpyKind_t memcpyDefault = ::hipMemcpyDefault;
        static constexpr MemcpyKind_t memcpyDeviceToDevice = ::hipMemcpyDeviceToDevice;
        static constexpr MemcpyKind_t memcpyDeviceToHost = ::hipMemcpyDeviceToHost;
        static constexpr MemcpyKind_t memcpyHostToDevice = ::hipMemcpyHostToDevice;

        static constexpr Flag_t streamDefault = hipStreamDefault;
        static constexpr Flag_t streamNonBlocking = hipStreamNonBlocking;

        static constexpr DeviceAttr_t deviceAttributeMaxBlockDimX = ::hipDeviceAttributeMaxBlockDimX;
        static constexpr DeviceAttr_t deviceAttributeMaxBlockDimY = ::hipDeviceAttributeMaxBlockDimY;
        static constexpr DeviceAttr_t deviceAttributeMaxBlockDimZ = ::hipDeviceAttributeMaxBlockDimZ;
        static constexpr DeviceAttr_t deviceAttributeMaxGridDimX = ::hipDeviceAttributeMaxGridDimX;
        static constexpr DeviceAttr_t deviceAttributeMaxGridDimY = ::hipDeviceAttributeMaxGridDimY;
        static constexpr DeviceAttr_t deviceAttributeMaxGridDimZ = ::hipDeviceAttributeMaxGridDimZ;
        static constexpr DeviceAttr_t deviceAttributeMaxSharedMemoryPerBlock
            = ::hipDeviceAttributeMaxSharedMemoryPerBlock;
        static constexpr DeviceAttr_t deviceAttributeMaxThreadsPerBlock = ::hipDeviceAttributeMaxThreadsPerBlock;
        static constexpr DeviceAttr_t deviceAttributeMultiprocessorCount = ::hipDeviceAttributeMultiprocessorCount;
        static constexpr DeviceAttr_t deviceAttributeWarpSize = ::hipDeviceAttributeWarpSize;

#    if HIP_VERSION >= 40'500'000
        static constexpr Limit_t limitPrintfFifoSize = ::hipLimitPrintfFifoSize;
#    else
        static constexpr Limit_t limitPrintfFifoSize
            = static_cast<Limit_t>(0x01); // Implemented only in ROCm 4.5.0 and later.
#    endif
        static constexpr Limit_t limitMallocHeapSize = ::hipLimitMallocHeapSize;

        // Host function helper
        // Encapsulates the different function signatures used by hipStreamAddCallback and hipLaunchHostFn, and the
        // different calling conventions used by CUDA (__stdcall on Win32) and HIP (standard).
        struct HostFnAdaptor
        {
            HostFn_t func_;
            void* data_;

            static void hostFunction(void* data)
            {
                auto ptr = reinterpret_cast<HostFnAdaptor*>(data);
                ptr->func_(ptr->data_);
                delete ptr;
            }

            static void streamCallback(Stream_t, Error_t, void* data)
            {
                auto ptr = reinterpret_cast<HostFnAdaptor*>(data);
                ptr->func_(ptr->data_);
                delete ptr;
            }
        };

        // Runtime API
        static inline Error_t deviceGetAttribute(int* value, DeviceAttr_t attr, int device)
        {
            return ::hipDeviceGetAttribute(value, attr, device);
        }

        static inline Error_t deviceGetLimit(size_t* pValue, Limit_t limit)
        {
#    if HIP_VERSION < 40'500'000
            if(limit == limitPrintfFifoSize)
            {
                // Implemented only in ROCm 4.5.0 and later.
                return errorUnsupportedLimit;
            }
#    endif
            return ::hipDeviceGetLimit(pValue, limit);
        }

        static inline Error_t deviceReset()
        {
            return ::hipDeviceReset();
        }

        static inline Error_t deviceSetLimit(Limit_t /* limit */, size_t /* value */)
        {
            // Not implemented.
            return errorUnsupportedLimit;
        }

        static inline Error_t deviceSynchronize()
        {
            return ::hipDeviceSynchronize();
        }

        static inline Error_t eventCreate(Event_t* event)
        {
            return ::hipEventCreate(event);
        }

        static inline Error_t eventCreateWithFlags(Event_t* event, Flag_t flags)
        {
            return ::hipEventCreateWithFlags(event, flags);
        }

        static inline Error_t eventDestroy(Event_t event)
        {
            return ::hipEventDestroy(event);
        }

        static inline Error_t eventQuery(Event_t event)
        {
            return ::hipEventQuery(event);
        }

        static inline Error_t eventRecord(Event_t event, Stream_t stream)
        {
            return ::hipEventRecord(event, stream);
        }

        static inline Error_t eventSynchronize(Event_t event)
        {
            return ::hipEventSynchronize(event);
        }

        static inline Error_t free(void* devPtr)
        {
            return ::hipFree(devPtr);
        }

        static inline Error_t freeAsync([[maybe_unused]] void* devPtr, [[maybe_unused]] Stream_t stream)
        {
            // stream-ordered memory operations are fully implemented only in ROCm 5.3.0 and later.
#    if HIP_VERSION >= 50'300'000
            // hipFreeAsync fails on a null pointer deallocation
            if(devPtr)
            {
                return ::hipFreeAsync(devPtr, stream);
            }
            else
            {
                return ::hipSuccess;
            }
#    else
            // Not implemented.
            return errorUnknown;
#    endif
        }

        static inline Error_t funcGetAttributes(FuncAttributes_t* attr, void const* func)
        {
            return ::hipFuncGetAttributes(attr, func);
        }

        template<typename T>
        static inline Error_t funcGetAttributes(FuncAttributes_t* attr, T* func)
        {
            return ::hipFuncGetAttributes(attr, reinterpret_cast<void const*>(func));
        }

        static inline Error_t getDeviceCount(int* count)
        {
            return ::hipGetDeviceCount(count);
        }

        static inline Error_t getDeviceProperties(DeviceProp_t* prop, int device)
        {
            return ::hipGetDeviceProperties(prop, device);
        }

        static inline char const* getErrorName(Error_t error)
        {
            return ::hipGetErrorName(error);
        }

        static inline char const* getErrorString(Error_t error)
        {
            return ::hipGetErrorString(error);
        }

        static inline Error_t getLastError()
        {
            return ::hipGetLastError();
        }

        static inline Error_t getSymbolAddress(void** devPtr, void const* symbol)
        {
            return ::hipGetSymbolAddress(devPtr, symbol);
        }

        template<class T>
        static inline Error_t getSymbolAddress(void** devPtr, T const& symbol)
        {
            return ::hipGetSymbolAddress(devPtr, symbol);
        }

        static inline Error_t hostGetDevicePointer(void** pDevice, void* pHost, Flag_t flags)
        {
            return ::hipHostGetDevicePointer(pDevice, pHost, flags);
        }

        static inline Error_t hostFree(void* ptr)
        {
            return ::hipHostFree(ptr);
        }

        static inline Error_t hostMalloc(void** ptr, size_t size, Flag_t flags)
        {
            return ::hipHostMalloc(ptr, size, flags);
        }

        static inline Error_t hostRegister(void* ptr, size_t size, Flag_t flags)
        {
            return ::hipHostRegister(ptr, size, flags);
        }

        static inline Error_t hostUnregister(void* ptr)
        {
            return ::hipHostUnregister(ptr);
        }

        static inline Error_t launchHostFunc(Stream_t stream, HostFn_t fn, void* userData)
        {
            // hipLaunchHostFunc is implemented only in ROCm 5.4.0 and later.
#    if HIP_VERSION >= 50'400'000
            // Wrap the host function using the proper calling convention.
            return ::hipLaunchHostFunc(stream, HostFnAdaptor::hostFunction, new HostFnAdaptor{fn, userData});
#    else
            // Emulate hipLaunchHostFunc using hipStreamAddCallback with a callback adaptor.
            return ::hipStreamAddCallback(stream, HostFnAdaptor::streamCallback, new HostFnAdaptor{fn, userData}, 0);
#    endif
        }

        static inline Error_t malloc(void** devPtr, size_t size)
        {
            return ::hipMalloc(devPtr, size);
        }

        static inline Error_t malloc3D(PitchedPtr_t* pitchedDevPtr, Extent_t extent)
        {
            return ::hipMalloc3D(pitchedDevPtr, extent);
        }

        static inline Error_t mallocAsync(
            [[maybe_unused]] void** devPtr,
            [[maybe_unused]] size_t size,
            [[maybe_unused]] Stream_t stream)
        {
            // stream-ordered memory operations are fully implemented only in ROCm 5.3.0 and later.
#    if HIP_VERSION >= 50'600'000
            return ::hipMallocAsync(devPtr, size, stream);
#    elif HIP_VERSION >= 50'300'000
            // before ROCm 5.6.0, hipMallocAsync fails for an allocation of 0 bytes
            if(size > 0)
            {
                return ::hipMallocAsync(devPtr, size, stream);
            }
            else
            {
                // make sure the pointer can safely be passed to hipFreeAsync
                *devPtr = nullptr;
                return ::hipSuccess;
            }
#    else
            // Not implemented.
            return errorUnknown;
#    endif
        }

        static inline Error_t mallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
        {
            return ::hipMallocPitch(devPtr, pitch, width, height);
        }

        static inline Error_t memGetInfo(size_t* free, size_t* total)
        {
            return ::hipMemGetInfo(free, total);
        }

        static inline Error_t memcpy(void* dst, void const* src, size_t count, MemcpyKind_t kind)
        {
            return ::hipMemcpy(dst, src, count, kind);
        }

        static inline Error_t memcpy2DAsync(
            void* dst,
            size_t dpitch,
            void const* src,
            size_t spitch,
            size_t width,
            size_t height,
            MemcpyKind_t kind,
            Stream_t stream)
        {
            return ::hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
        }

        static inline Error_t memcpy3DAsync(Memcpy3DParms_t const* p, Stream_t stream)
        {
            return ::hipMemcpy3DAsync(p, stream);
        }

        static inline Error_t memcpyAsync(void* dst, void const* src, size_t count, MemcpyKind_t kind, Stream_t stream)
        {
            return ::hipMemcpyAsync(dst, src, count, kind, stream);
        }

        static inline Error_t memset2DAsync(
            void* devPtr,
            size_t pitch,
            int value,
            size_t width,
            size_t height,
            Stream_t stream)
        {
            return ::hipMemset2DAsync(devPtr, pitch, value, width, height, stream);
        }

        static inline Error_t memset3DAsync(PitchedPtr_t pitchedDevPtr, int value, Extent_t extent, Stream_t stream)
        {
            return ::hipMemset3DAsync(pitchedDevPtr, value, extent, stream);
        }

        static inline Error_t memsetAsync(void* devPtr, int value, size_t count, Stream_t stream)
        {
            return ::hipMemsetAsync(devPtr, value, count, stream);
        }

        static inline Error_t setDevice(int device)
        {
            return ::hipSetDevice(device);
        }

        static inline Error_t streamCreate(Stream_t* pStream)
        {
            return ::hipStreamCreate(pStream);
        }

        static inline Error_t streamCreateWithFlags(Stream_t* pStream, Flag_t flags)
        {
            return ::hipStreamCreateWithFlags(pStream, flags);
        }

        static inline Error_t streamDestroy(Stream_t stream)
        {
            return ::hipStreamDestroy(stream);
        }

        static inline Error_t streamQuery(Stream_t stream)
        {
            return ::hipStreamQuery(stream);
        }

        static inline Error_t streamSynchronize(Stream_t stream)
        {
            return ::hipStreamSynchronize(stream);
        }

        static inline Error_t streamWaitEvent(Stream_t stream, Event_t event, Flag_t flags)
        {
            return ::hipStreamWaitEvent(stream, event, flags);
        }

        static inline PitchedPtr_t makePitchedPtr(void* d, size_t p, size_t xsz, size_t ysz)
        {
            return ::make_hipPitchedPtr(d, p, xsz, ysz);
        }

        static inline Pos_t makePos(size_t x, size_t y, size_t z)
        {
            return ::make_hipPos(x, y, z);
        }

        static inline Extent_t makeExtent(size_t w, size_t h, size_t d)
        {
            return ::make_hipExtent(w, h, d);
        }
    };

} // namespace alpaka

#endif // ALPAKA_ACC_GPU_HIP_ENABLED
