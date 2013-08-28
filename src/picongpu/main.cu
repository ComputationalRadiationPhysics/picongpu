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
#include "particles/factories/CoverTypes.hpp"
#include "math/MapTuple.hpp"
#include "particles/memory/frames/Frame.hpp"
#include "particles/memory/boxes/TileDataBox.hpp"

#include <iostream>

using namespace PMacc;

namespace PMacc{
identifier( double, position, 0.0 );
identifier( bool, a, true );
identifier( float, b, 1.0 );
identifier( bool, c, false );
}

//using namespace picongpu;

__global__ void kernel(int* in,int imax)
{
    typedef bmpl::map <
    bmpl::pair<position, position::type>, // Key, Value Paare
    bmpl::pair<a, int>,
    bmpl::pair<b, float>
    > particle;
    
    typedef Frame<CastToVector, particle> FrameType;
    
    FrameType x;
    x.getIdentifier( a_).z( ) = 1.11f;
    for(int i=0;i<imax;++i)
        x.getIdentifier( a_).z( ) += 22;
    
    *in=x.getIdentifier( a_).z( );
}

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
    typedef bmpl::map <
        bmpl::pair<position, position::type>, // Key, Value Paare
        bmpl::pair<a, int>,
        bmpl::pair<b, float>
        > particle;

    //typedef typename CoverTypes<typename particle::Map, CastToVector>::type VectorParticle;
    // typedef math::MapTuple <VectorParticle> Frame;

    typedef Frame<CastToVector, particle> FrameType;

    FrameType x;
    // b d;
    //  x(b())=VectorDataBox<float>(new float(10));
    x.getIdentifier(a_).z( ) = 2.11f;
    x[2][a_] = 11;
    x[2][a_] = x[2][a_]+ 1;


    std::cout << "sizeof=" << sizeof (FrameType ) << std::endl;
    std::cout << "value=" << x.getIdentifier(b_).x( ) << std::endl;
    std::cout << "value=" << x[2][a_] << " -"<<traits::HasIdentifier<typename FrameType::ParticleType,c>::value<<"-"<<std::endl;

    return 0;
}


