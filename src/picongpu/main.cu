/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, René Widera
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

/**
 * @mainpage PIConGPU-Frame
 *
 * Project with HZDR for porting their PiC-code to a GPU cluster.
 *
 * \image html picongpu.jpg
 *
 * @author Heiko Burau, Rene Widera, Wolfgang Hoenig, Felix Schmitt, Axel Huebl, Michael Bussmann, Guido Juckeland
 */


/*#include <simulation_defines.hpp>
#include <mpi.h>
#include "communication/manager_common.h"
 */

#include  "types.h"
#include "particles/factories/CreateMap.hpp"
#include "math/MapTuple.hpp"
#include "particles/memory/frames/Frame.hpp"
#include "particles/memory/boxes/TileDataBox.hpp"
#include <boost/mpl/list.hpp>
#include <boost/mpl/vector.hpp>
#include "algorithms/ForEach.hpp"

#include <iostream>

#include "RefWrapper.hpp"

using namespace PMacc;
using namespace PMacc::algorithms::forEach;

namespace PMacc
{

}

identifier(hallo);

//using namespace picongpu;

/*
__global__ void kernel( int* in, int imax )
{

    typedef bmpl::vector <
        position,
        a,
        c
        > particle;

    typedef Frame<CastToVector, particle> FrameType;

    FrameType x;


    x.getIdentifier( a_ ).z( ) = 1.11f;
    x.getIdentifier( a_ ).z( ) += a::getDefaultValue();
    for ( int i = 0; i < imax; ++i )
        x.getIdentifier( a_ ).z( ) += 22;
    x.getIdentifier( a_ ).z( ) += 2;

 *in = x.getIdentifier( a_ ).z( );
}


 */

template<typename T, typename Type>
struct MallocMemory
{

    template<typename ValueType1 >
            HDINLINE void operator()(RefWrapper<ValueType1> v1, const size_t size) const
    {
        v1.get().getIdentifier(T()) = VectorDataBox<Type>(new Type[size]);
    }
};

template<typename T>
struct SetDefault
{

    template<typename ValueType1 >
            HDINLINE void operator()(RefWrapper<ValueType1> v1, const size_t size) const
    {
        for (size_t i = 0; i < size; ++i)
            v1.get().getIdentifier(T())[i] = T::defaultValue;
    }
};

template<typename T>
struct SetDefaultValue
{

    template<typename ValueType1 >
            HDINLINE void operator()(RefWrapper<ValueType1> v1) const
    {
        v1.get()[T()] = T::getDefaultValue();
    }
};

template<typename T>
struct FreeMemory
{

    template<typename ValueType1 >
            HDINLINE void operator()(RefWrapper<ValueType1> v1) const
    {
        delete[] v1().getIdentifier(T()).getPointer();
    }
};


value_identifier(double, position, 0.0);
value_identifier(int, aInt, 42);
value_identifier(double, aDouble, 42.1);
value_identifier(float, b, 1.0);
value_identifier(bool, c, false);
alias(horst);


int main(int argc, char **argv)
{

    typedef bmpl::vector <
            position,
            horst<aDouble>,
            b
            > particle;

    typedef Frame<CastToArray, particle> FrameType;
    FrameType frame;
    PMACC_AUTO(par, frame[255]);
    ForEach<particle, SetDefaultValue<void> >()(byRef(par));
    std::cout << "sizeof=" << sizeof (FrameType) << std::endl;

    std::cout << "value=" << frame[255][horst_] << " -" << std::endl;
    std::cout << traits::HasIdentifier<typename FrameType::ParticleType, horst<aDouble> >::value << "- true" << std::endl;
    std::cout << traits::HasIdentifier<typename FrameType::ParticleType, horst<> >::value << "- true" << std::endl;
    std::cout << traits::HasIdentifier<typename FrameType::ParticleType, horst<aInt> >::value << "- false" << std::endl;
    
    std::cout << traits::hasIdentifier(frame,horst<aDouble>()) << "- true" << std::endl;
    std::cout << traits::hasIdentifier(frame,horst_) << "- true" << std::endl;
    std::cout << traits::hasIdentifier(frame,horst<aInt>()) << "- false" << std::endl;
    //ForEach<MemList, FreeMemory<void> > freemem;
    //freemem( byRef(x) );

    return 0;
}


