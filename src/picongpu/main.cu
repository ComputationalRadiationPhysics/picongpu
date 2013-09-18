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
#include "particles/factories/CreateIdentifierMap.hpp"
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
identifier( double, position, 0.0 );
identifier( int, a, 42 );
identifier( float, b, 1.0 );
identifier( bool, c, false );

}

wildcard( hallo );

//using namespace picongpu;

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
    x.getIdentifier( a_ ).z( ) += 11;
    for ( int i = 0; i < imax; ++i )
        x.getIdentifier( a_ ).z( ) += 22;
    x.getIdentifier( a_ ).z( ) += 2;

    *in = x.getIdentifier( a_ ).z( );
}




template<typename T,typename Type>
struct MallocMemory
{

    template<typename ValueType1 >
        HDINLINE void operator( )( RefWrapper<ValueType1> v1, const size_t size) const
    {
        v1.get().getIdentifier( T( ) ) = VectorDataBox<Type>( new Type[size] );
    }
};

template<typename T>
struct SetDefault
{
    template<typename ValueType1 >
        HDINLINE void operator( )( RefWrapper<ValueType1> v1, const size_t size) const
    {
        for ( size_t i = 0; i < size; ++i )
            v1.get().getIdentifier( T( ) )[i] = T::defaultValue;
    }
};

template<typename T>
struct SetDefaultValue
{

    template<typename ValueType1 >
        HDINLINE void operator( )( RefWrapper<ValueType1> v1 ) const
    {
        v1.get()[T( )] = T::defaultValue;
    }
};

template<typename T>
struct FreeMemory
{

    template<typename ValueType1 >
        HDINLINE void operator( )( RefWrapper<ValueType1> v1 ) const
    {
        delete[] v1().getIdentifier( T( ) ).getPointer( );
    }
};

/*! start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main( int argc, char **argv )
{
    // using namespace host;
    /*MPI_CHECK( MPI_Init( &argc, &argv ) );

    picongpu::simulation_starter::SimStarter sim;
    if ( !sim.parseConfigs( argc, argv ) )
    {
        MPI_CHECK( MPI_Finalize( ) );
        return 1;
    }

    sim.load( );
    sim.start( );
    sim.unload( );

    MPI_CHECK( MPI_Finalize( ) );*/




    // typedef math::MapTuple <
    typedef bmpl::vector <
        position,
        a,
        b
        > particle;

    //typedef typename CoverTypes<typename particle::Map, CastToVector>::type VectorParticle;
    // typedef math::MapTuple <VectorParticle> Frame;

    typedef Frame<CastToVectorBox, particle> FrameType;

    FrameType x;

    typedef bmpl::list<a> MemList;

    ForEach<MemList, MallocMemory<void,int> > alloc;
   // size_t size = 100 * 1024 * 1024;
    alloc( byRef(x), 100 * 1024 * 1024 );
    PMACC_AUTO( par, x[100 * 1024 * 1024 - 1] );
    ForEach<MemList, SetDefaultValue<void> >()( byRef(par) );

   // printf( "nach: %X\n", x.getIdentifier( a_ ).getPointer( ) );

    //x[100 * 1024 * 1024 - 1][a_] = 11;
    //PMACC_AUTO( par, x[100 * 1024 * 1024 - 1] );
    //par[a_] = par[a_] + 1;


    std::cout << "sizeof=" << sizeof (FrameType ) << std::endl;
    //    std::cout << "value=" << x.getIdentifier(b_).x( ) << std::endl;
    std::cout << "value=" << x[100 * 1024 * 1024 - 1][a_] << " -" << traits::HasIdentifier<FrameType, position>::value << "-" << std::endl;

    ForEach<MemList, FreeMemory<void> > freemem;
    freemem( byRef(x) );

    return 0;
}


