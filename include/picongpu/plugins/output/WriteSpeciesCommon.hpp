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


    template<typename T_Type>
    struct MallocMappedMemory
    {
        template<typename ValueType>
        HINLINE void operator()(ValueType& v1, const size_t size) const
        {
            using type = typename pmacc::traits::Resolve<T_Type>::type::type;

            bool isMappedMemorySupported = alpaka::hasMappedBufSupport<::alpaka::Pltf<cupla::AccDev>>;

            PMACC_VERIFY_MSG(isMappedMemorySupported, "Device must support mapped memory!");

            type* ptr = nullptr;
            if(size != 0)
            {
                // Memory is automatically mapped to the device if supported.
                CUDA_CHECK(cuplaMallocHost((void**) &ptr, size * sizeof(type)));
            }
            v1.getIdentifier(T_Type()) = VectorDataBox<type>(ptr);
        }
    };

    /** allocate memory on host
     *
     * This functor use `new[]` to allocate memory
     */
    template<typename T_Attribute>
    struct MallocHostMemory
    {
        template<typename ValueType>
        HINLINE void operator()(ValueType& v1, const size_t size) const
        {
            using Attribute = T_Attribute;
            using type = typename pmacc::traits::Resolve<Attribute>::type::type;

            type* ptr = nullptr;
            if(size != 0)
            {
                ptr = new type[size];
            }
            v1.getIdentifier(Attribute()) = VectorDataBox<type>(ptr);
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

    template<typename T_Type>
    struct FreeMappedMemory
    {
        template<typename ValueType>
        HINLINE void operator()(ValueType& value) const
        {
            auto* ptr = value.getIdentifier(T_Type()).getPointer();
            if(ptr != nullptr)
            {
                CUDA_CHECK(cuplaFreeHost(ptr));
            }
        }
    };

    //! Free memory
    template<typename T_Attribute>
    struct FreeHostMemory
    {
        template<typename ValueType>
        HINLINE void operator()(ValueType& value) const
        {
            using Attribute = T_Attribute;

            auto* ptr = value.getIdentifier(Attribute()).getPointer();
            delete[] ptr;
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

} // namespace picongpu
