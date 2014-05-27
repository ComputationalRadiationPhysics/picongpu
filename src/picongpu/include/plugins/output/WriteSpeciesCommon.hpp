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

#include "RefWrapper.hpp"
#include <boost/type_traits.hpp>

#include "mappings/kernel/AreaMapping.hpp"

#include "compileTime/conversion/RemoveFromSeq.hpp"

namespace picongpu
{

using namespace PMacc;



template<typename T_Type>
struct MallocMemory
{
    template<typename ValueType >
    HINLINE void operator()(RefWrapper<ValueType> v1, const size_t size) const
    {
        typedef typename T_Type::type type;

        type* ptr = NULL;
        if (size != 0)
        {
            CUDA_CHECK(cudaHostAlloc(&ptr, size * sizeof (type), cudaHostAllocMapped));
        }
        v1.get().getIdentifier(T_Type()) = VectorDataBox<type>(ptr);

    }
};

template<typename T_Type>
struct GetDevicePtr
{
    template<typename ValueType >
    HINLINE void operator()(RefWrapper<ValueType> dest, RefWrapper<ValueType> src) const
    {
        typedef typename T_Type::type type;

        type* ptr = NULL;
        type* srcPtr = src.get().getIdentifier(T_Type()).getPointer();
        if (srcPtr != NULL)
        {
            CUDA_CHECK(cudaHostGetDevicePointer(&ptr, srcPtr, 0));
        }
        dest.get().getIdentifier(T_Type()) =
            VectorDataBox<type>(ptr);
    }
};

template<typename T_Type>
struct FreeMemory
    {
    template<typename ValueType >
    HINLINE void operator()(RefWrapper<ValueType> value) const
    {
        typedef typename T_Type::type type;

        type* ptr = value.get().getIdentifier(T_Type()).getPointer();
        if (ptr != NULL)
            CUDA_CHECK(cudaFreeHost(ptr));
    }
};

/*functor to create a pair for a MapTupel map*/
struct OperatorCreateVectorBox
{
    template<typename InType>
    struct apply
    {
        typedef
        bmpl::pair< InType,
        PMacc::VectorDataBox< typename InType::type > >
        type;
    };
};

} //namespace picongpu

