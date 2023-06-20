/* Copyright 2014-2022 Rene Widera, Felix Schmitt
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu
{
    using namespace pmacc;

    template<typename Frame>
    void mallocMappedFrameMemory(Frame& frame)
    {
        constexpr bool isMappedMemorySupported = alpaka::hasMappedBufSupport<::alpaka::Pltf<cupla::AccDev>>;
        PMACC_VERIFY_MSG(isMappedMemorySupported, "Device must support mapped memory!");

        int i = 0;
        for(std::byte*& ptr : frame.blobs())
        {
            const auto size = frame.blobSize(i);
            if(size != 0)
            {
                // Memory is automatically mapped to the device if supported.
                CUDA_CHECK(cuplaMallocHost((void**) &ptr, size));
            }
            else
                ptr = nullptr;
            log<picLog::INPUT_OUTPUT>("openPMD:     blob %1%, size: %2%, ptr: %3%") % i % size % ptr;
            i++;
        }
    }

    /** allocate memory on host
     *
     * This functor use `new[]` to allocate memory
     */
    template<typename Frame>
    void mallocFrameMemory(Frame& frame)
    {
        constexpr bool isMappedMemorySupported = alpaka::hasMappedBufSupport<::alpaka::Pltf<cupla::AccDev>>;
        PMACC_VERIFY_MSG(isMappedMemorySupported, "Device must support mapped memory!");

        int i = 0;
        for(std::byte*& ptr : frame.blobs())
        {
            const auto size = frame.blobSize(i);
            if(size != 0)
                ptr = new std::byte[size];
            else
                ptr = nullptr;
            log<picLog::INPUT_OUTPUT>("openPMD:     blob %1%, size: %2%, ptr: %3%") % i % size % ptr;
            i++;
        }
    }

    /** copy species to host memory
     *
     * use `DataConnector::get<...>()` to copy data
     */
    template<typename T_SpeciesType>
    struct CopySpeciesToHost
    {
        typedef T_SpeciesType SpeciesType;

        HINLINE void operator()() const
        {
            /* DataConnector copies data to host */
            DataConnector& dc = Environment<>::get().DataConnector();
            auto dataSet = dc.get<SpeciesType>(SpeciesType::FrameType::getName());
            dataSet->synchronize();
        }
    };

    template<typename Frame>
    void freeMappedFrameMemory(Frame& frame)
    {
        for(std::byte*& ptr : frame.blobs())
            if(ptr != nullptr)
            {
                CUDA_CHECK(cuplaFreeHost(ptr));
                ptr = nullptr;
            }
    }

    template<typename Frame>
    void freeFrameMemory(Frame& frame)
    {
        for(auto* ptr : frame.blobs())
        {
            delete[] ptr;
            ptr = nullptr;
        }
    }
} // namespace picongpu
