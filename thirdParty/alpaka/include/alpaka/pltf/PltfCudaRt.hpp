/**
* \file
* Copyright 2014-2015 Benjamin Worpitz, Rene Widera
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevCudaRt.hpp>

#include <alpaka/core/Cuda.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace alpaka
{
    namespace pltf
    {
        //#############################################################################
        //! The CUDA RT device manager.
        class PltfCudaRt
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST PltfCudaRt() = delete;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device manager device type trait specialization.
            template<>
            struct DevType<
                pltf::PltfCudaRt>
            {
                using type = dev::DevCudaRt;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU platform device count get trait specialization.
            template<>
            struct GetDevCount<
                pltf::PltfCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDevCount()
                -> std::size_t
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    int iNumDevices(0);
                    cudaError_t error = cudaGetDeviceCount(&iNumDevices);
                    if(error != cudaSuccess)
                        iNumDevices = 0;

                    return static_cast<std::size_t>(iNumDevices);
                }
            };

            //#############################################################################
            //! The CPU platform device get trait specialization.
            template<>
            struct GetDevByIdx<
                pltf::PltfCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDevByIdx(
                    std::size_t const & devIdx)
                -> dev::DevCudaRt
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    dev::DevCudaRt dev;

                    std::size_t const devCount(pltf::getDevCount<pltf::PltfCudaRt>());
                    if(devIdx >= devCount)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << devIdx << ". There are only " << devCount << " CUDA devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    if(isDevUsable(devIdx))
                    {
                        dev.m_iDevice = static_cast<int>(devIdx);

                        // Log this device.
    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        cudaDeviceProp devProp;
                        ALPAKA_CUDA_RT_CHECK(cudaGetDeviceProperties(&devProp, dev.m_iDevice));
    #endif
    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        printDeviceProperties(devProp);
    #elif ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        std::cout << BOOST_CURRENT_FUNCTION << devProp.name << std::endl;
    #endif
                    }
                    else
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << devIdx << ". It is not accessible!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return dev;
                }

            private:
                //-----------------------------------------------------------------------------
                //! \return If the device is usable.
                ALPAKA_FN_HOST static auto isDevUsable(
                    std::size_t iDevice)
                -> bool
                {
                    cudaError rc(cudaSetDevice(static_cast<int>(iDevice)));

                    cudaStream_t stream = {};
                    // Create a dummy stream to check if the device is already used by an other process.
                    // cudaSetDevice never returns an error if another process already uses the selected device and gpu compute mode is set "process exclusive".
                    // \TODO: Check if this workaround is needed!
                    if(rc == cudaSuccess)
                    {
                        rc = cudaStreamCreate(&stream);
                    }

                    if(rc == cudaSuccess)
                    {
                        // Destroy the dummy stream.
                        ALPAKA_CUDA_RT_CHECK(
                            cudaStreamDestroy(
                                stream));
                        return true;
                    }
                    else
                    {
                        // Return the previous error from cudaStreamCreate.
                        ALPAKA_CUDA_RT_CHECK(
                            rc);
                        // Reset the Error state.
                        cudaGetLastError();

                        return false;
                    }
                }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                //-----------------------------------------------------------------------------
                //! Prints all the device properties to std::cout.
                ALPAKA_FN_HOST static auto printDeviceProperties(
                    cudaDeviceProp const & devProp)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    std::size_t const kiB(1024);
                    std::size_t const miB(kiB * kiB);
                    std::cout << "name: " << devProp.name << std::endl;
                    std::cout << "totalGlobalMem: " << devProp.totalGlobalMem/miB << " MiB" << std::endl;
                    std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock/kiB << " KiB" << std::endl;
                    std::cout << "regsPerBlock: " << devProp.regsPerBlock << std::endl;
                    std::cout << "warpSize: " << devProp.warpSize << std::endl;
                    std::cout << "memPitch: " << devProp.memPitch << " B" << std::endl;
                    std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
                    std::cout << "maxThreadsDim[3]: (" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")" << std::endl;
                    std::cout << "maxGridSize[3]: (" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << ")" << std::endl;
                    std::cout << "clockRate: " << devProp.clockRate << " kHz" << std::endl;
                    std::cout << "totalConstMem: " << devProp.totalConstMem/kiB << " KiB" << std::endl;
                    std::cout << "major: " << devProp.major << std::endl;
                    std::cout << "minor: " << devProp.minor << std::endl;
                    std::cout << "textureAlignment: " << devProp.textureAlignment << std::endl;
                    std::cout << "texturePitchAlignment: " << devProp.texturePitchAlignment << std::endl;
                    //std::cout << "deviceOverlap: " << devProp.deviceOverlap << std::endl;    // Deprecated
                    std::cout << "multiProcessorCount: " << devProp.multiProcessorCount << std::endl;
                    std::cout << "kernelExecTimeoutEnabled: " << devProp.kernelExecTimeoutEnabled << std::endl;
                    std::cout << "integrated: " << devProp.integrated << std::endl;
                    std::cout << "canMapHostMemory: " << devProp.canMapHostMemory << std::endl;
                    std::cout << "computeMode: " << devProp.computeMode << std::endl;
                    std::cout << "maxTexture1D: " << devProp.maxTexture1D << std::endl;
                    std::cout << "maxTexture1DLinear: " << devProp.maxTexture1DLinear << std::endl;
                    std::cout << "maxTexture2D[2]: " << devProp.maxTexture2D[0] << "x" << devProp.maxTexture2D[1] << std::endl;
                    std::cout << "maxTexture2DLinear[3]: " << devProp.maxTexture2DLinear[0] << "x" << devProp.maxTexture2DLinear[1] << "x" << devProp.maxTexture2DLinear[2] << std::endl;
                    std::cout << "maxTexture2DGather[2]: " << devProp.maxTexture2DGather[0] << "x" << devProp.maxTexture2DGather[1] << std::endl;
                    std::cout << "maxTexture3D[3]: " << devProp.maxTexture3D[0] << "x" << devProp.maxTexture3D[1] << "x" << devProp.maxTexture3D[2] << std::endl;
                    std::cout << "maxTextureCubemap: " << devProp.maxTextureCubemap << std::endl;
                    std::cout << "maxTexture1DLayered[2]: " << devProp.maxTexture1DLayered[0] << "x" << devProp.maxTexture1DLayered[1] << std::endl;
                    std::cout << "maxTexture2DLayered[3]: " << devProp.maxTexture2DLayered[0] << "x" << devProp.maxTexture2DLayered[1] << "x" << devProp.maxTexture2DLayered[2] << std::endl;
                    std::cout << "maxTextureCubemapLayered[2]: " << devProp.maxTextureCubemapLayered[0] << "x" << devProp.maxTextureCubemapLayered[1] << std::endl;
                    std::cout << "maxSurface1D: " << devProp.maxSurface1D << std::endl;
                    std::cout << "maxSurface2D[2]: " << devProp.maxSurface2D[0] << "x" << devProp.maxSurface2D[1] << std::endl;
                    std::cout << "maxSurface3D[3]: " << devProp.maxSurface3D[0] << "x" << devProp.maxSurface3D[1] << "x" << devProp.maxSurface3D[2] << std::endl;
                    std::cout << "maxSurface1DLayered[2]: " << devProp.maxSurface1DLayered[0] << "x" << devProp.maxSurface1DLayered[1] << std::endl;
                    std::cout << "maxSurface2DLayered[3]: " << devProp.maxSurface2DLayered[0] << "x" << devProp.maxSurface2DLayered[1] << "x" << devProp.maxSurface2DLayered[2] << std::endl;
                    std::cout << "maxSurfaceCubemap: " << devProp.maxSurfaceCubemap << std::endl;
                    std::cout << "maxSurfaceCubemapLayered[2]: " << devProp.maxSurfaceCubemapLayered[0] << "x" << devProp.maxSurfaceCubemapLayered[1] << std::endl;
                    std::cout << "surfaceAlignment: " << devProp.surfaceAlignment << std::endl;
                    std::cout << "concurrentKernels: " << devProp.concurrentKernels << std::endl;
                    std::cout << "ECCEnabled: " << devProp.ECCEnabled << std::endl;
                    std::cout << "pciBusID: " << devProp.pciBusID << std::endl;
                    std::cout << "pciDeviceID: " << devProp.pciDeviceID << std::endl;
                    std::cout << "pciDomainID: " << devProp.pciDomainID << std::endl;
                    std::cout << "tccDriver: " << devProp.tccDriver << std::endl;
                    std::cout << "asyncEngineCount: " << devProp.asyncEngineCount << std::endl;
                    std::cout << "unifiedAddressing: " << devProp.unifiedAddressing << std::endl;
                    std::cout << "memoryClockRate: " << devProp.memoryClockRate << " kHz" << std::endl;
                    std::cout << "memoryBusWidth: " << devProp.memoryBusWidth << " b" << std::endl;
                    std::cout << "l2CacheSize: " << devProp.l2CacheSize << " B" << std::endl;
                    std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
                }
#endif
            };
        }
    }
}

#endif
