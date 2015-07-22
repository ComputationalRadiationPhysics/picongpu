/**
 * Copyright 2014 Rene Widera, Felix Schmitt
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

#include "types.h"
#include "simulation_types.hpp"

#include "plugins/ISimulationPlugin.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include "compileTime/conversion/MakeSeq.hpp"

#include <boost/type_traits.hpp>

#include "mappings/kernel/AreaMapping.hpp"

#include "compileTime/conversion/RemoveFromSeq.hpp"
#include "traits/Resolve.hpp"

namespace picongpu
{

using namespace PMacc;



template<typename T_Type>
struct MallocMemory
{
    template<typename ValueType >
    HINLINE void operator()(ValueType& v1, const size_t size) const
    {
        typedef typename PMacc::traits::Resolve<T_Type>::type::type type;

        type* ptr = NULL;
        if (size != 0)
        {
            CUDA_CHECK(cudaHostAlloc(&ptr, size * sizeof (type), cudaHostAllocMapped));
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
    template<typename ValueType >
    HINLINE void operator()(ValueType& v1, const size_t size) const
    {
        typedef T_Attribute Attribute;
        typedef typename PMacc::traits::Resolve<Attribute>::type::type type;

        type* ptr = NULL;
        if (size != 0)
        {
            ptr = new type[size];
        }
        v1.getIdentifier(Attribute()) = VectorDataBox<type>(ptr);

    }
};


/** copy species to host memory
 *
 * use `DataConnector::getData<...>()` to copy data
 */
template<typename T_SpeciesType>
struct CopySpeciesToHost
{
    typedef T_SpeciesType SpeciesType;

    HINLINE void operator()() const
    {
        /* DataConnector copies data to host */
        DataConnector &dc = Environment<>::get().DataConnector();
        dc.getData<SpeciesType> (SpeciesType::FrameType::getName());
        dc.releaseData(SpeciesType::FrameType::getName());
    }
};

template<typename T_Type>
struct GetDevicePtr
{
    template<typename ValueType >
    HINLINE void operator()(ValueType& dest, ValueType& src)
    {
        typedef typename PMacc::traits::Resolve<T_Type>::type::type type;

        type* ptr = NULL;
        type* srcPtr = src.getIdentifier(T_Type()).getPointer();
        if (srcPtr != NULL)
        {
            CUDA_CHECK(cudaHostGetDevicePointer(&ptr, srcPtr, 0));
        }
        dest.getIdentifier(T_Type()) = VectorDataBox<type>(ptr);
    }
};

template<typename T_Type>
struct FreeMemory
{
    template<typename ValueType >
    HINLINE void operator()(ValueType& value) const
    {
        typedef typename PMacc::traits::Resolve<T_Type>::type::type type;

        type* ptr = value.getIdentifier(T_Type()).getPointer();
        if (ptr != NULL)
        {
            CUDA_CHECK(cudaFreeHost(ptr));
            ptr=NULL;
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

    template<typename ValueType >
    HINLINE void operator()(ValueType& value) const
    {
        typedef T_Attribute Attribute;
        typedef typename PMacc::traits::Resolve<Attribute>::type::type type;

        type* ptr = value.getIdentifier(Attribute()).getPointer();
        if (ptr != NULL)
        {
            __deleteArray(ptr);
            ptr=NULL;
        }
    }
};

/*functor to create a pair for a MapTuple map*/
struct OperatorCreateVectorBox
{
    template<typename InType>
    struct apply
    {
        typedef
        bmpl::pair< InType,
        PMacc::VectorDataBox< typename PMacc::traits::Resolve<InType>::type::type > >
        type;
    };
};

} //namespace picongpu

