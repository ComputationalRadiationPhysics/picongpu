/* Copyright 2022 Benjamin Worpitz, René Widera, Andrea Bocci, Bernhard Manfred Gruber, Antonio Di Pilato,
 *                Christian Kaever
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/dev/Traits.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka
{
    // Forward declarations.
    struct ApiCudaRt;
    struct ApiHipRt;

    //! The CUDA/HIP RT platform.
    template<typename TApi>
    struct PlatformUniformCudaHipRt : concepts::Implements<ConceptPlatform, PlatformUniformCudaHipRt<TApi>>
    {
#    if defined(BOOST_COMP_GNUC) && BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(11, 0, 0)                                 \
        && BOOST_COMP_GNUC < BOOST_VERSION_NUMBER(12, 0, 0)
        // This is a workaround for g++-11 bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96295
        // g++-11 complains in *all* places where a PlatformCpu is used, that it "may be used uninitialized"
        char c = {};
#    endif
    };

    namespace trait
    {
        //! The CUDA/HIP RT platform device type trait specialization.
        template<typename TApi>
        struct DevType<PlatformUniformCudaHipRt<TApi>>
        {
            using type = DevUniformCudaHipRt<TApi>;
        };

        //! The CUDA/HIP RT platform device count get trait specialization.
        template<typename TApi>
        struct GetDevCount<PlatformUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getDevCount(PlatformUniformCudaHipRt<TApi> const&) -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                int iNumDevices(0);
                typename TApi::Error_t error = TApi::getDeviceCount(&iNumDevices);
                if(error != TApi::success)
                    iNumDevices = 0;

                return static_cast<std::size_t>(iNumDevices);
            }
        };

        //! The CUDA/HIP RT platform device get trait specialization.
        template<typename TApi>
        struct GetDevByIdx<PlatformUniformCudaHipRt<TApi>>
        {
            ALPAKA_FN_HOST static auto getDevByIdx(
                PlatformUniformCudaHipRt<TApi> const& platform,
                std::size_t const& devIdx) -> DevUniformCudaHipRt<TApi>
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                std::size_t const devCount = getDevCount(platform);
                if(devIdx >= devCount)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for device " << devIdx << ". There are only " << devCount
                          << " devices!";
                    throw std::runtime_error(ssErr.str());
                }

                if(isDevUsable(devIdx))
                {
                    DevUniformCudaHipRt<TApi> dev(static_cast<int>(devIdx));

                    // Log this device.
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    typename TApi::DeviceProp_t devProp;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getDeviceProperties(&devProp, dev.getNativeHandle()));
#    endif
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    printDeviceProperties(devProp);
#    elif ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    std::cout << __func__ << devProp.name << std::endl;
#    endif
                    return dev;
                }
                else
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for device " << devIdx << ". It is not accessible!";
                    throw std::runtime_error(ssErr.str());
                }
            }

        private:
            //! \return If the device is usable.
            ALPAKA_FN_HOST static auto isDevUsable(std::size_t iDevice) -> bool
            {
                typename TApi::Error_t rc = TApi::setDevice(static_cast<int>(iDevice));
                typename TApi::Stream_t queue = {};
                // Create a dummy queue to check if the device is already used by an other process.
                // cuda/hip-SetDevice never returns an error if another process already uses the selected device and
                // gpu compute mode is set "process exclusive". \TODO: Check if this workaround is needed!
                if(rc == TApi::success)
                {
                    rc = TApi::streamCreate(&queue);
                }

                if(rc == TApi::success)
                {
                    // Destroy the dummy queue.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamDestroy(queue));
                    return true;
                }
                else
                {
                    // Return the previous error from cudaStreamCreate.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(rc);
                    // Reset the Error state.
                    std::ignore = TApi::getLastError();
                    return false;
                }
            }

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            //! Prints all the device properties to std::cout.
            ALPAKA_FN_HOST static auto printDeviceProperties(typename TApi::DeviceProp_t const& devProp) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                constexpr auto KiB = std::size_t{1024};
                constexpr auto MiB = KiB * KiB;
                std::cout << "name: " << devProp.name << std::endl;
                std::cout << "totalGlobalMem: " << devProp.totalGlobalMem / MiB << " MiB" << std::endl;
                std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock / KiB << " KiB" << std::endl;
                std::cout << "regsPerBlock: " << devProp.regsPerBlock << std::endl;
                std::cout << "warpSize: " << devProp.warpSize << std::endl;
                std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
                std::cout << "maxThreadsDim[3]: (" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1]
                          << ", " << devProp.maxThreadsDim[2] << ")" << std::endl;
                std::cout << "maxGridSize[3]: (" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", "
                          << devProp.maxGridSize[2] << ")" << std::endl;
                std::cout << "clockRate: " << devProp.clockRate << " kHz" << std::endl;
                std::cout << "totalConstMem: " << devProp.totalConstMem / KiB << " KiB" << std::endl;
                std::cout << "major: " << devProp.major << std::endl;
                std::cout << "minor: " << devProp.minor << std::endl;

                // std::cout << "deviceOverlap: " << devProp.deviceOverlap << std::endl;    // Deprecated
                std::cout << "multiProcessorCount: " << devProp.multiProcessorCount << std::endl;
                std::cout << "integrated: " << devProp.integrated << std::endl;
                std::cout << "canMapHostMemory: " << devProp.canMapHostMemory << std::endl;
                std::cout << "computeMode: " << devProp.computeMode << std::endl;
                std::cout << "concurrentKernels: " << devProp.concurrentKernels << std::endl;
                std::cout << "pciBusID: " << devProp.pciBusID << std::endl;
                std::cout << "pciDeviceID: " << devProp.pciDeviceID << std::endl;
                std::cout << "pciDomainID: " << devProp.pciDomainID << std::endl;
                std::cout << "memoryClockRate: " << devProp.memoryClockRate << " kHz" << std::endl;
                std::cout << "memoryBusWidth: " << devProp.memoryBusWidth << " b" << std::endl;
                std::cout << "l2CacheSize: " << devProp.l2CacheSize << " B" << std::endl;
                std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
                std::cout << "isMultiGpuBoard: " << devProp.isMultiGpuBoard << std::endl;
                if constexpr(std::is_same_v<TApi, ApiCudaRt>)
                {
                    std::cout << "memPitch: " << devProp.memPitch << " B" << std::endl;
                    std::cout << "textureAlignment: " << devProp.textureAlignment << std::endl;
                    std::cout << "texturePitchAlignment: " << devProp.texturePitchAlignment << std::endl;
                    std::cout << "kernelExecTimeoutEnabled: " << devProp.kernelExecTimeoutEnabled << std::endl;
                    std::cout << "unifiedAddressing: " << devProp.unifiedAddressing << std::endl;
                    std::cout << "multiGpuBoardGroupID: " << devProp.multiGpuBoardGroupID << std::endl;
                    std::cout << "singleToDoublePrecisionPerfRatio: " << devProp.singleToDoublePrecisionPerfRatio
                              << std::endl;
                    std::cout << "pageableMemoryAccess: " << devProp.pageableMemoryAccess << std::endl;
                    std::cout << "concurrentManagedAccess: " << devProp.concurrentManagedAccess << std::endl;
                    std::cout << "computePreemptionSupported: " << devProp.computePreemptionSupported << std::endl;
                    std::cout << "canUseHostPointerForRegisteredMem: " << devProp.canUseHostPointerForRegisteredMem
                              << std::endl;
                    std::cout << "cooperativeLaunch: " << devProp.cooperativeLaunch << std::endl;
                    std::cout << "cooperativeMultiDeviceLaunch: " << devProp.cooperativeMultiDeviceLaunch << std::endl;
                    std::cout << "maxTexture1D: " << devProp.maxTexture1D << std::endl;
                    std::cout << "maxTexture1DLinear: " << devProp.maxTexture1DLinear << std::endl;
                    std::cout << "maxTexture2D[2]: " << devProp.maxTexture2D[0] << "x" << devProp.maxTexture2D[1]
                              << std::endl;
                    std::cout << "maxTexture2DLinear[3]: " << devProp.maxTexture2DLinear[0] << "x"
                              << devProp.maxTexture2DLinear[1] << "x" << devProp.maxTexture2DLinear[2] << std::endl;
                    std::cout << "maxTexture2DGather[2]: " << devProp.maxTexture2DGather[0] << "x"
                              << devProp.maxTexture2DGather[1] << std::endl;
                    std::cout << "maxTexture3D[3]: " << devProp.maxTexture3D[0] << "x" << devProp.maxTexture3D[1]
                              << "x" << devProp.maxTexture3D[2] << std::endl;
                    std::cout << "maxTextureCubemap: " << devProp.maxTextureCubemap << std::endl;
                    std::cout << "maxTexture1DLayered[2]: " << devProp.maxTexture1DLayered[0] << "x"
                              << devProp.maxTexture1DLayered[1] << std::endl;
                    std::cout << "maxTexture2DLayered[3]: " << devProp.maxTexture2DLayered[0] << "x"
                              << devProp.maxTexture2DLayered[1] << "x" << devProp.maxTexture2DLayered[2] << std::endl;
                    std::cout << "maxTextureCubemapLayered[2]: " << devProp.maxTextureCubemapLayered[0] << "x"
                              << devProp.maxTextureCubemapLayered[1] << std::endl;
                    std::cout << "maxSurface1D: " << devProp.maxSurface1D << std::endl;
                    std::cout << "maxSurface2D[2]: " << devProp.maxSurface2D[0] << "x" << devProp.maxSurface2D[1]
                              << std::endl;
                    std::cout << "maxSurface3D[3]: " << devProp.maxSurface3D[0] << "x" << devProp.maxSurface3D[1]
                              << "x" << devProp.maxSurface3D[2] << std::endl;
                    std::cout << "maxSurface1DLayered[2]: " << devProp.maxSurface1DLayered[0] << "x"
                              << devProp.maxSurface1DLayered[1] << std::endl;
                    std::cout << "maxSurface2DLayered[3]: " << devProp.maxSurface2DLayered[0] << "x"
                              << devProp.maxSurface2DLayered[1] << "x" << devProp.maxSurface2DLayered[2] << std::endl;
                    std::cout << "maxSurfaceCubemap: " << devProp.maxSurfaceCubemap << std::endl;
                    std::cout << "maxSurfaceCubemapLayered[2]: " << devProp.maxSurfaceCubemapLayered[0] << "x"
                              << devProp.maxSurfaceCubemapLayered[1] << std::endl;
                    std::cout << "surfaceAlignment: " << devProp.surfaceAlignment << std::endl;
                    std::cout << "ECCEnabled: " << devProp.ECCEnabled << std::endl;
                    std::cout << "tccDriver: " << devProp.tccDriver << std::endl;
                    std::cout << "asyncEngineCount: " << devProp.asyncEngineCount << std::endl;
                    std::cout << "streamPrioritiesSupported: " << devProp.streamPrioritiesSupported << std::endl;
                    std::cout << "globalL1CacheSupported: " << devProp.globalL1CacheSupported << std::endl;
                    std::cout << "localL1CacheSupported: " << devProp.localL1CacheSupported << std::endl;
                    std::cout << "sharedMemPerMultiprocessor: " << devProp.sharedMemPerMultiprocessor << std::endl;
                    std::cout << "regsPerMultiprocessor: " << devProp.regsPerMultiprocessor << std::endl;
                    std::cout << "managedMemory: " << devProp.managedMemory << std::endl;
                }
                else
                { // ApiHipRt
                    std::cout << "clockInstructionRate: " << devProp.clockInstructionRate << "kHz" << std::endl;
                    std::cout << "maxSharedMemoryPerMultiProcessor: " << devProp.maxSharedMemoryPerMultiProcessor / KiB
                              << " KiB" << std::endl;
                    std::cout << "gcnArch: " << devProp.gcnArch << std::endl;
                    std::cout << "arch: " << std::endl;
                    std::cout << "    hasGlobalInt32Atomics: " << devProp.arch.hasGlobalInt32Atomics << std::endl;
                    std::cout << "    hasGlobalFloatAtomicExch: " << devProp.arch.hasGlobalFloatAtomicExch
                              << std::endl;
                    std::cout << "    hasSharedInt32Atomics: " << devProp.arch.hasSharedInt32Atomics << std::endl;
                    std::cout << "    hasSharedFloatAtomicExch: " << devProp.arch.hasSharedFloatAtomicExch
                              << std::endl;
                    std::cout << "    hasFloatAtomicAdd: " << devProp.arch.hasFloatAtomicAdd << std::endl;
                    std::cout << "    hasGlobalInt64Atomics: " << devProp.arch.hasGlobalInt64Atomics << std::endl;
                    std::cout << "    hasSharedInt64Atomics: " << devProp.arch.hasSharedInt64Atomics << std::endl;
                    std::cout << "    hasDoubles: " << devProp.arch.hasDoubles << std::endl;
                    std::cout << "    hasWarpVote: " << devProp.arch.hasWarpVote << std::endl;
                    std::cout << "    hasWarpBallot: " << devProp.arch.hasWarpBallot << std::endl;
                    std::cout << "    hasWarpShuffle: " << devProp.arch.hasWarpShuffle << std::endl;
                    std::cout << "    hasFunnelShift: " << devProp.arch.hasFunnelShift << std::endl;
                    std::cout << "    hasThreadFenceSystem: " << devProp.arch.hasThreadFenceSystem << std::endl;
                    std::cout << "    hasSyncThreadsExt: " << devProp.arch.hasSyncThreadsExt << std::endl;
                    std::cout << "    hasSurfaceFuncs: " << devProp.arch.hasSurfaceFuncs << std::endl;
                    std::cout << "    has3dGrid: " << devProp.arch.has3dGrid << std::endl;
                    std::cout << "    hasDynamicParallelism: " << devProp.arch.hasDynamicParallelism << std::endl;
                }
            }
#    endif
        };
    } // namespace trait
} // namespace alpaka

#endif
