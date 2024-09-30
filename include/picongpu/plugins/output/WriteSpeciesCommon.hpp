/* Copyright 2014-2023 Rene Widera, Felix Schmitt
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/Pair.hpp>
#include <pmacc/particles/memory/boxes/TileDataBox.hpp>
#include <pmacc/traits/Resolve.hpp>

namespace picongpu
{
    using namespace pmacc;


    template<typename T_Attribute>
    struct MallocMappedMemory
    {
        template<typename T_Buffers, typename T_Frame>
        HINLINE void operator()(T_Buffers& buffers, T_Frame& frame, size_t const size) const
        {
            using type = typename pmacc::traits::Resolve<T_Attribute>::type::type;

            bool isMappedMemorySupported = alpaka::hasMappedBufSupport<::alpaka::Platform<pmacc::ComputeDevice>>;

            PMACC_VERIFY_MSG(isMappedMemorySupported, "Device must support mapped memory!");

            frame.getIdentifier(T_Attribute()) = nullptr;
            if(size != 0)
            {
                buffers.getIdentifier(T_Attribute()) = alpaka::allocMappedBufIfSupported<type, MemIdxType>(
                    manager::Device<HostDevice>::get().current(),
                    manager::Device<ComputeDevice>::get().getPlatform(),
                    MemSpace<DIM1>(size).toAlpakaMemVec());
                frame.getIdentifier(T_Attribute()) = alpaka::getPtrNative(*buffers.getIdentifier(T_Attribute()));
            }
        }
    };

    /** allocate memory on host
     *
     * This functor use `new[]` to allocate memory
     */
    template<typename T_Attribute>
    struct MallocHostMemory
    {
        template<typename T_Buffers, typename T_Frame>
        HINLINE void operator()(T_Buffers& buffers, T_Frame& frame, const size_t size) const
        {
            using Attribute = T_Attribute;
            using type = typename pmacc::traits::Resolve<Attribute>::type::type;

            frame.getIdentifier(T_Attribute()) = nullptr;
            if(size != 0)
            {
                buffers.getIdentifier(T_Attribute()) = alpaka::allocBuf<type, MemIdxType>(
                    manager::Device<HostDevice>::get().current(),
                    MemSpace<DIM1>(size).toAlpakaMemVec());
                frame.getIdentifier(T_Attribute()) = alpaka::getPtrNative(*buffers.getIdentifier(T_Attribute()));
            }
        }
    };


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

    /*functor to create a pair for a MapTuple map*/
    struct OperatorCreateVectorBox
    {
        template<typename InType>
        struct apply
        {
            using type
                = pmacc::meta::Pair<InType, pmacc::VectorDataBox<typename pmacc::traits::Resolve<InType>::type::type>>;
        };
    };

    struct OperatorCreateAlpakaBuffer
    {
        template<typename InType>
        struct apply
        {
            using ValueType = typename pmacc::traits::Resolve<InType>::type::type;
            using BufferType = ::alpaka::Buf<HostDevice, ValueType, AlpakaDim<DIM1>, MemIdxType>;
            using type = pmacc::meta::Pair<InType, std::optional<BufferType>>;
        };
    };

} // namespace picongpu
