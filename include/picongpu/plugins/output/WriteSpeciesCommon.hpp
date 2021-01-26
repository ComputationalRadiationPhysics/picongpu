/* Copyright 2014-2021 Rene Widera, Felix Schmitt
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

#include "picongpu/plugins/ISimulationPlugin.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/type_traits.hpp>


namespace picongpu
{
    using namespace pmacc;


    template<typename T_Type>
    struct MallocMemory
    {
        template<typename ValueType>
        HINLINE void operator()(ValueType& v1, const size_t size) const
        {
            typedef typename pmacc::traits::Resolve<T_Type>::type::type type;

            type* ptr = nullptr;
            if(size != 0)
            {
#if(PMACC_CUDA_ENABLED == 1)
                CUDA_CHECK((cuplaError_t) cudaHostAlloc(&ptr, size * sizeof(type), cudaHostAllocMapped));
#elif(ALPAKA_ACC_GPU_HIP_ENABLED == 1)
                CUDA_CHECK((cuplaError_t) hipHostMalloc((void**) &ptr, size * sizeof(type), hipHostRegisterMapped));
#else
                ptr = new type[size];
#endif
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
            typedef T_Attribute Attribute;
            typedef typename pmacc::traits::Resolve<Attribute>::type::type type;

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
            dc.get<SpeciesType>(SpeciesType::FrameType::getName());
            dc.releaseData(SpeciesType::FrameType::getName());
        }
    };

    template<typename T_Type>
    struct GetDevicePtr
    {
        template<typename ValueType>
        HINLINE void operator()(ValueType& dest, ValueType& src)
        {
            typedef typename pmacc::traits::Resolve<T_Type>::type::type type;

            type* ptr = nullptr;
            type* srcPtr = src.getIdentifier(T_Type()).getPointer();
            if(srcPtr != nullptr)
            {
#if(PMACC_CUDA_ENABLED == 1)
                CUDA_CHECK((cuplaError_t) cudaHostGetDevicePointer(&ptr, srcPtr, 0));
#elif(ALPAKA_ACC_GPU_HIP_ENABLED == 1)
                CUDA_CHECK((cuplaError_t) hipHostGetDevicePointer((void**) &ptr, srcPtr, 0));
#else
                ptr = srcPtr;
#endif
            }
            dest.getIdentifier(T_Type()) = VectorDataBox<type>(ptr);
        }
    };

    template<typename T_Type>
    struct FreeMemory
    {
        template<typename ValueType>
        HINLINE void operator()(ValueType& value) const
        {
            typedef typename pmacc::traits::Resolve<T_Type>::type::type type;

            type* ptr = value.getIdentifier(T_Type()).getPointer();
            if(ptr != nullptr)
            {
#if(PMACC_CUDA_ENABLED == 1)
/* cupla 0.2.0 does not support the function cudaHostAlloc to create mapped memory.
 * Therefore we need to call the native CUDA function cudaFreeHost to free memory.
 * Due to the renaming of cuda functions with cupla via macros we need to remove
 * the renaming to get access to the native cuda function.
 * @todo this is a workaround please fix me. We need to investigate if
 * it is possible to have mapped/unified memory in alpaka.
 *
 * corresponding alpaka issues:
 *   https://github.com/ComputationalRadiationPhysics/alpaka/issues/296
 *   https://github.com/ComputationalRadiationPhysics/alpaka/issues/612
 */
#    undef cudaFreeHost
                CUDA_CHECK((cuplaError_t) cudaFreeHost(ptr));
// re-introduce the cupla macro
#    define cudaFreeHost(...) cuplaFreeHost(__VA_ARGS__)
#elif(ALPAKA_ACC_GPU_HIP_ENABLED == 1)
                CUDA_CHECK((cuplaError_t) hipHostFree(ptr));
#else
                __deleteArray(ptr);
#endif
            }
        }
    };

    /** free memory
     *
     * use `__deleteArray()` to free memory
     */
    template<typename T_Attribute>
    struct FreeHostMemory
    {
        template<typename ValueType>
        HINLINE void operator()(ValueType& value) const
        {
            typedef T_Attribute Attribute;
            typedef typename pmacc::traits::Resolve<Attribute>::type::type type;

            type* ptr = value.getIdentifier(Attribute()).getPointer();
            if(ptr != nullptr)
            {
                __deleteArray(ptr);
                ptr = nullptr;
            }
        }
    };

    /*functor to create a pair for a MapTuple map*/
    struct OperatorCreateVectorBox
    {
        template<typename InType>
        struct apply
        {
            typedef bmpl::pair<InType, pmacc::VectorDataBox<typename pmacc::traits::Resolve<InType>::type::type>> type;
        };
    };

} // namespace picongpu
