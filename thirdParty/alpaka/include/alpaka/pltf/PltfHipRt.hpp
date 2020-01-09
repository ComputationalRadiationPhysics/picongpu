/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Concepts.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevHipRt.hpp>

#include <alpaka/core/Hip.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace alpaka
{
    namespace pltf
    {
        //#############################################################################
        //! The HIP RT device manager.
        class PltfHipRt :
            public concepts::Implements<ConceptPltf, PltfHipRt>
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            ALPAKA_FN_HOST PltfHipRt() = delete;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT device manager device type trait specialization.
            template<>
            struct DevType<
                pltf::PltfHipRt>
            {
                using type = dev::DevHipRt;
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
                pltf::PltfHipRt>
            {
                //-----------------------------------------------------------------------------

                ALPAKA_FN_HOST static auto getDevCount()
                -> std::size_t
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    int iNumDevices(0);
                    hipError_t error = hipGetDeviceCount(&iNumDevices);
                    if(error != hipSuccess)
                        iNumDevices = 0;
                    return static_cast<std::size_t>(iNumDevices);
                }
            };

            //#############################################################################
            //! The CPU platform device get trait specialization.
            template<>
            struct GetDevByIdx<
                pltf::PltfHipRt>
            {
                //-----------------------------------------------------------------------------

                ALPAKA_FN_HOST static auto getDevByIdx(
                    std::size_t const & devIdx)
                -> dev::DevHipRt
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    dev::DevHipRt dev;

                    std::size_t const devCount(pltf::getDevCount<pltf::PltfHipRt>());
                    if(devIdx >= devCount)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << devIdx << ". There are only " << devCount << " HIP devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    if(isDevUsable(devIdx))
                    {
                        dev.m_iDevice = static_cast<int>(devIdx);

                        // Log this device.
    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        hipDeviceProp_t devProp;
                        ALPAKA_HIP_RT_CHECK(hipGetDeviceProperties(&devProp, dev.m_iDevice));
    #endif
    #if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        printDeviceProperties(devProp);
    #elif ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                        std::cout << __func__ << devProp.name << std::endl;
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
                    hipError_t rc(hipSetDevice(static_cast<int>(iDevice)));

                    hipStream_t queue = {};
                    // Create a dummy queue to check if the device is already used by an other process.
                    // hipSetDevice never returns an error if another process already uses the selected device and gpu compute mode is set "process exclusive".
                    // \TODO: Check if this workaround is needed!
                    if(rc == hipSuccess)
                    {
                        rc = hipStreamCreate(&queue);
                    }

                    if(rc == hipSuccess)
                    {
                        // Destroy the dummy queue.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamDestroy(
                                queue));
                        return true;
                    }
                    else
                    {
                        // Return the previous error from hipStreamCreate.
                        ALPAKA_HIP_RT_CHECK(
                            rc);
                        // Reset the Error state.
                        hipGetLastError();

                        return false;
                    }
                }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                //-----------------------------------------------------------------------------
                //! Prints all the device properties to std::cout.
                ALPAKA_FN_HOST static auto printDeviceProperties(
                    hipDeviceProp_t const & devProp)
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
                    std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
                    std::cout << "maxThreadsDim[3]: (" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")" << std::endl;
                    std::cout << "maxGridSize[3]: (" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << ")" << std::endl;
                    std::cout << "clockRate: " << devProp.clockRate << " kHz" << std::endl;
                    std::cout << "memoryClockRate: " << devProp.memoryClockRate << " kHz" << std::endl;
                    std::cout << "memoryBusWidth: " << devProp.memoryBusWidth << " b" << std::endl;
                    std::cout << "totalConstMem: " << devProp.totalConstMem/kiB << " KiB" << std::endl;
                    std::cout << "major: " << devProp.major << std::endl;
                    std::cout << "minor: " << devProp.minor << std::endl;
                    std::cout << "multiProcessorCount: " << devProp.multiProcessorCount << std::endl;
                    std::cout << "l2CacheSize: " << devProp.l2CacheSize << " B" << std::endl;
                    std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
                    std::cout << "computeMode: " << devProp.computeMode << std::endl;
                    std::cout << "clockInstructionRate: " << devProp.clockInstructionRate << "kHz" << std::endl;
                    std::cout << "arch: " << std::endl;
                    std::cout << "    hasGlobalInt32Atomics: " << devProp.arch.hasGlobalInt32Atomics << std::endl;
                    std::cout << "    hasGlobalFloatAtomicExch: " << devProp.arch.hasGlobalFloatAtomicExch << std::endl;
                    std::cout << "    hasSharedInt32Atomics: " << devProp.arch.hasSharedInt32Atomics << std::endl;
                    std::cout << "    hasSharedFloatAtomicExch: " << devProp.arch.hasSharedFloatAtomicExch << std::endl;
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
                    std::cout << "concurrentKernels: " << devProp.concurrentKernels << std::endl;
                    std::cout << "pciDomainID: " << devProp.pciDomainID << std::endl;
                    std::cout << "pciBusID: " << devProp.pciBusID << std::endl;
                    std::cout << "pciDeviceID: " << devProp.pciDeviceID << std::endl;
                    std::cout << "maxSharedMemoryPerMultiProcessor: " << devProp.maxSharedMemoryPerMultiProcessor/kiB << " KiB" << std::endl;
                    std::cout << "isMultiGpuBoard: " << devProp.isMultiGpuBoard << std::endl;
                    std::cout << "canMapHostMemory: " << devProp.canMapHostMemory << std::endl;
                    std::cout << "gcnArch: " << devProp.gcnArch << std::endl;
                    std::cout << "integrated: " << devProp.integrated << std::endl;
                }
#endif
            };
        }
    }
}

#endif
