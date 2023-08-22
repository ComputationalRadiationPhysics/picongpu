/* Copyright 2022 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <boost/predef.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#    include <cuda_runtime_api.h>

namespace alpaka
{
    struct ApiCudaRt
    {
        // Names
        static constexpr char name[] = "Cuda";
        static constexpr auto version = BOOST_PREDEF_MAKE_10_VVRRP(CUDART_VERSION);

        // Types
        using DeviceAttr_t = ::cudaDeviceAttr;
        using DeviceProp_t = ::cudaDeviceProp;
        using Error_t = ::cudaError_t;
        using Event_t = ::cudaEvent_t;
        using Extent_t = ::cudaExtent;
        using Flag_t = unsigned int;
        using FuncAttributes_t = ::cudaFuncAttributes;
        using HostFn_t = void (*)(void* data); // same as cudaHostFn_t, without the CUDART_CB calling convention
        using Limit_t = ::cudaLimit;
        using Memcpy3DParms_t = ::cudaMemcpy3DParms;
        using MemcpyKind_t = ::cudaMemcpyKind;
        using PitchedPtr_t = ::cudaPitchedPtr;
        using Pos_t = ::cudaPos;
        using Stream_t = ::cudaStream_t;

        // Constants
        static constexpr Error_t success = ::cudaSuccess;
        static constexpr Error_t errorNotReady = ::cudaErrorNotReady;
        static constexpr Error_t errorHostMemoryAlreadyRegistered = ::cudaErrorHostMemoryAlreadyRegistered;
        static constexpr Error_t errorHostMemoryNotRegistered = ::cudaErrorHostMemoryNotRegistered;
        static constexpr Error_t errorUnsupportedLimit = ::cudaErrorUnsupportedLimit;
        static constexpr Error_t errorUnknown = ::cudaErrorUnknown;

        static constexpr Flag_t eventDefault = cudaEventDefault;
        static constexpr Flag_t eventBlockingSync = cudaEventBlockingSync;
        static constexpr Flag_t eventDisableTiming = cudaEventDisableTiming;
        static constexpr Flag_t eventInterprocess = cudaEventInterprocess;

        static constexpr Flag_t hostMallocDefault = cudaHostAllocDefault;
        static constexpr Flag_t hostMallocMapped = cudaHostAllocMapped;
        static constexpr Flag_t hostMallocPortable = cudaHostAllocPortable;
        static constexpr Flag_t hostMallocWriteCombined = cudaHostAllocWriteCombined;
        static constexpr Flag_t hostMallocCoherent = cudaHostAllocDefault; // Not supported.
        static constexpr Flag_t hostMallocNonCoherent = cudaHostAllocDefault; // Not supported.

        static constexpr Flag_t hostRegisterDefault = cudaHostRegisterDefault;
        static constexpr Flag_t hostRegisterPortable = cudaHostRegisterPortable;
        static constexpr Flag_t hostRegisterMapped = cudaHostRegisterMapped;
        static constexpr Flag_t hostRegisterIoMemory = cudaHostRegisterIoMemory;

        static constexpr MemcpyKind_t memcpyDefault = ::cudaMemcpyDefault;
        static constexpr MemcpyKind_t memcpyDeviceToDevice = ::cudaMemcpyDeviceToDevice;
        static constexpr MemcpyKind_t memcpyDeviceToHost = ::cudaMemcpyDeviceToHost;
        static constexpr MemcpyKind_t memcpyHostToDevice = ::cudaMemcpyHostToDevice;

        static constexpr Flag_t streamDefault = cudaStreamDefault;
        static constexpr Flag_t streamNonBlocking = cudaStreamNonBlocking;

        static constexpr DeviceAttr_t deviceAttributeMaxBlockDimX = ::cudaDevAttrMaxBlockDimX;
        static constexpr DeviceAttr_t deviceAttributeMaxBlockDimY = ::cudaDevAttrMaxBlockDimY;
        static constexpr DeviceAttr_t deviceAttributeMaxBlockDimZ = ::cudaDevAttrMaxBlockDimZ;
        static constexpr DeviceAttr_t deviceAttributeMaxGridDimX = ::cudaDevAttrMaxGridDimX;
        static constexpr DeviceAttr_t deviceAttributeMaxGridDimY = ::cudaDevAttrMaxGridDimY;
        static constexpr DeviceAttr_t deviceAttributeMaxGridDimZ = ::cudaDevAttrMaxGridDimZ;
        static constexpr DeviceAttr_t deviceAttributeMaxSharedMemoryPerBlock = ::cudaDevAttrMaxSharedMemoryPerBlock;
        static constexpr DeviceAttr_t deviceAttributeMaxThreadsPerBlock = ::cudaDevAttrMaxThreadsPerBlock;
        static constexpr DeviceAttr_t deviceAttributeMultiprocessorCount = ::cudaDevAttrMultiProcessorCount;

        static constexpr Limit_t limitPrintfFifoSize = ::cudaLimitPrintfFifoSize;
        static constexpr Limit_t limitMallocHeapSize = ::cudaLimitMallocHeapSize;

        // Host function helper
        // Encapsulates the different function signatures used by cudaStreamAddCallback and cudaLaunchHostFn, and the
        // different calling conventions used by CUDA (__stdcall on Win32) and HIP (standard).
        struct HostFnAdaptor
        {
            HostFn_t func_;
            void* data_;

            static void CUDART_CB hostFunction(void* data)
            {
                auto ptr = reinterpret_cast<HostFnAdaptor*>(data);
                ptr->func_(ptr->data_);
                delete ptr;
            }

            static void CUDART_CB streamCallback(Stream_t, Error_t, void* data)
            {
                auto ptr = reinterpret_cast<HostFnAdaptor*>(data);
                ptr->func_(ptr->data_);
                delete ptr;
            }
        };

        // Runtime API
        static inline Error_t deviceGetAttribute(int* value, DeviceAttr_t attr, int device)
        {
            return ::cudaDeviceGetAttribute(value, attr, device);
        }

        static inline Error_t deviceGetLimit(size_t* pValue, Limit_t limit)
        {
            return ::cudaDeviceGetLimit(pValue, limit);
        }

        static inline Error_t deviceReset()
        {
            return ::cudaDeviceReset();
        }

        static inline Error_t deviceSetLimit(Limit_t limit, size_t value)
        {
            return ::cudaDeviceSetLimit(limit, value);
        }

        static inline Error_t deviceSynchronize()
        {
            return ::cudaDeviceSynchronize();
        }

        static inline Error_t eventCreate(Event_t* event)
        {
            return ::cudaEventCreate(event);
        }

        static inline Error_t eventCreateWithFlags(Event_t* event, Flag_t flags)
        {
            return ::cudaEventCreateWithFlags(event, flags);
        }

        static inline Error_t eventDestroy(Event_t event)
        {
            return ::cudaEventDestroy(event);
        }

        static inline Error_t eventQuery(Event_t event)
        {
            return ::cudaEventQuery(event);
        }

        static inline Error_t eventRecord(Event_t event, Stream_t stream)
        {
            return ::cudaEventRecord(event, stream);
        }

        static inline Error_t eventSynchronize(Event_t event)
        {
            return ::cudaEventSynchronize(event);
        }

        static inline Error_t free(void* devPtr)
        {
            return ::cudaFree(devPtr);
        }

        static inline Error_t freeAsync([[maybe_unused]] void* devPtr, [[maybe_unused]] Stream_t stream)
        {
#    if CUDART_VERSION >= 11020
            return ::cudaFreeAsync(devPtr, stream);
#    else
            // Not implemented.
            return errorUnknown;
#    endif
        }

        static inline Error_t funcGetAttributes(FuncAttributes_t* attr, void const* func)
        {
            return ::cudaFuncGetAttributes(attr, func);
        }

        template<typename T>
        static inline Error_t funcGetAttributes(FuncAttributes_t* attr, T* func)
        {
            return ::cudaFuncGetAttributes(attr, reinterpret_cast<void const*>(func));
        }

        static inline Error_t getDeviceCount(int* count)
        {
            return ::cudaGetDeviceCount(count);
        }

        static inline Error_t getDeviceProperties(DeviceProp_t* prop, int device)
        {
            return ::cudaGetDeviceProperties(prop, device);
        }

        static inline char const* getErrorName(Error_t error)
        {
            return ::cudaGetErrorName(error);
        }

        static inline char const* getErrorString(Error_t error)
        {
            return ::cudaGetErrorString(error);
        }

        static inline Error_t getLastError()
        {
            return ::cudaGetLastError();
        }

        static inline Error_t getSymbolAddress(void** devPtr, void const* symbol)
        {
            return ::cudaGetSymbolAddress(devPtr, symbol);
        }

        template<class T>
        static inline Error_t getSymbolAddress(void** devPtr, T const& symbol)
        {
            return ::cudaGetSymbolAddress(devPtr, symbol);
        }

        static inline Error_t hostGetDevicePointer(void** pDevice, void* pHost, Flag_t flags)
        {
            return ::cudaHostGetDevicePointer(pDevice, pHost, flags);
        }

        static inline Error_t hostFree(void* ptr)
        {
            return ::cudaFreeHost(ptr);
        }

        static inline Error_t hostMalloc(void** ptr, size_t size, Flag_t flags)
        {
            return ::cudaHostAlloc(ptr, size, flags);
        }

        static inline Error_t hostRegister(void* ptr, size_t size, Flag_t flags)
        {
            return ::cudaHostRegister(ptr, size, flags);
        }

        static inline Error_t hostUnregister(void* ptr)
        {
            return ::cudaHostUnregister(ptr);
        }

        static inline Error_t launchHostFunc(Stream_t stream, HostFn_t fn, void* userData)
        {
#    if CUDART_VERSION >= 10000
            // Wrap the host function using the proper calling convention
            return ::cudaLaunchHostFunc(stream, HostFnAdaptor::hostFunction, new HostFnAdaptor{fn, userData});
#    else
            // Emulate cudaLaunchHostFunc using cudaStreamAddCallback with a callback adaptor.
            return ::cudaStreamAddCallback(stream, HostFnAdaptor::streamCallback, new HostFnAdaptor{fn, userData}, 0);
#    endif
        }

        static inline Error_t malloc(void** devPtr, size_t size)
        {
            return ::cudaMalloc(devPtr, size);
        }

        static inline Error_t malloc3D(PitchedPtr_t* pitchedDevPtr, Extent_t extent)
        {
            return ::cudaMalloc3D(pitchedDevPtr, extent);
        }

        static inline Error_t mallocAsync(
            [[maybe_unused]] void** devPtr,
            [[maybe_unused]] size_t size,
            [[maybe_unused]] Stream_t stream)
        {
#    if CUDART_VERSION >= 11020
            return ::cudaMallocAsync(devPtr, size, stream);
#    else
            // Not implemented.
            return errorUnknown;
#    endif
        }

        static inline Error_t mallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
        {
            return ::cudaMallocPitch(devPtr, pitch, width, height);
        }

        static inline Error_t memGetInfo(size_t* free, size_t* total)
        {
            return ::cudaMemGetInfo(free, total);
        }

        static inline Error_t memcpy(void* dst, void const* src, size_t count, MemcpyKind_t kind)
        {
            return ::cudaMemcpy(dst, src, count, kind);
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
            return ::cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
        }

        static inline Error_t memcpy3DAsync(Memcpy3DParms_t const* p, Stream_t stream)
        {
            return ::cudaMemcpy3DAsync(p, stream);
        }

        static inline Error_t memcpyAsync(void* dst, void const* src, size_t count, MemcpyKind_t kind, Stream_t stream)
        {
            return ::cudaMemcpyAsync(dst, src, count, kind, stream);
        }

        static inline Error_t memset2DAsync(
            void* devPtr,
            size_t pitch,
            int value,
            size_t width,
            size_t height,
            Stream_t stream)
        {
            return ::cudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
        }

        static inline Error_t memset3DAsync(PitchedPtr_t pitchedDevPtr, int value, Extent_t extent, Stream_t stream)
        {
            return ::cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
        }

        static inline Error_t memsetAsync(void* devPtr, int value, size_t count, Stream_t stream)
        {
            return ::cudaMemsetAsync(devPtr, value, count, stream);
        }

        static inline Error_t setDevice(int device)
        {
            return ::cudaSetDevice(device);
        }

        static inline Error_t streamCreate(Stream_t* pStream)
        {
            return ::cudaStreamCreate(pStream);
        }

        static inline Error_t streamCreateWithFlags(Stream_t* pStream, Flag_t flags)
        {
            return ::cudaStreamCreateWithFlags(pStream, flags);
        }

        static inline Error_t streamDestroy(Stream_t stream)
        {
            return ::cudaStreamDestroy(stream);
        }

        static inline Error_t streamQuery(Stream_t stream)
        {
            return ::cudaStreamQuery(stream);
        }

        static inline Error_t streamSynchronize(Stream_t stream)
        {
            return ::cudaStreamSynchronize(stream);
        }

        static inline Error_t streamWaitEvent(Stream_t stream, Event_t event, Flag_t flags)
        {
            return ::cudaStreamWaitEvent(stream, event, flags);
        }

        static inline PitchedPtr_t makePitchedPtr(void* d, size_t p, size_t xsz, size_t ysz)
        {
            return ::make_cudaPitchedPtr(d, p, xsz, ysz);
        }

        static inline Pos_t makePos(size_t x, size_t y, size_t z)
        {
            return ::make_cudaPos(x, y, z);
        }

        static inline Extent_t makeExtent(size_t w, size_t h, size_t d)
        {
            return ::make_cudaExtent(w, h, d);
        }
    };

} // namespace alpaka

#endif // ALPAKA_ACC_GPU_CUDA_ENABLED
