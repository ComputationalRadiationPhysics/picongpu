/**
 * Copyright 2013 René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 
#include "types.hpp"
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include "Simulation.hpp"

#include <mpi.h>
#include "communication/manager_common.h"

namespace po = boost::program_options;

/*! start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main( int argc, char **argv )
{

    MPI_CHECK( MPI_Init( &argc, &argv ) );

    typedef ::gol::Space Space;

    std::vector<uint32_t> devices;
    std::vector<uint32_t> gridSize;
    std::vector<uint32_t> periodic;
    uint32_t steps;
    std::string rule;

    po::options_description desc( "Allowed options" );
    desc.add_options( )
            ( "help,h", "produce help message" )
            ( "steps,s", po::value<uint32_t > ( &steps ), "simulation steps" )
            ( "rule,r", po::value<std::string > ( &rule ), "simulation rule etc. 23/3" )
            ( "devices,d", po::value<std::vector<uint32_t> > ( &devices )->multitoken( ), "number of devices in each dimension" )
            ( "grid,g", po::value<std::vector<uint32_t> > ( &gridSize )->multitoken( ),
              "size of the simulation grid (real size maybe smaller because each GPU needs a border)" )
            ( "periodic", po::value<std::vector<uint32_t> > ( &periodic )->multitoken( ),
              "specifying whether the grid is periodic (1) or not (0) in each dimension, default: no periodic dimensions" );

    // parse command line options and config file and store values in vm
    po::variables_map vm;
    po::store( boost::program_options::parse_command_line( argc, argv, desc ), vm );
    po::notify( vm );

    // print help message and quit simulation
    if ( vm.count( "help" ) )
    {
        MPI_CHECK( MPI_Finalize( ) );
        std::cerr << desc << "\n";
        return false;
    }


    //fill periodic with 0
    while ( periodic.size( ) < DIM2 )
        periodic.push_back( 0 );

    // check on correct number of devices. fill with default value 1 for missing dimensions
    if ( devices.size( ) > DIM2 )
    {
        std::cerr << "Invalid number of devices.\nuse [-d dx=1 dy=1 dz=1]" << std::endl;
    }
    else
        while ( devices.size( ) < DIM2 )
            devices.push_back( 1 );

    // check on correct grid size. fill with default grid size value 1 for missing 3. dimension
    if ( gridSize.size( ) != DIM2 )
    {
        std::cerr << "Invalid or missing grid size.\nuse -g width height [depth=1]" << std::endl;
        MPI_CHECK( MPI_Finalize( ) );
        return 0;
    }


    Space gpus( devices[0], devices[1] );
    Space grid( gridSize[0], gridSize[1] );
    Space endless( periodic[0], periodic[1] );

    uint32_t ruleMask = 0;
    size_t strLen = rule.length( );
    size_t gPoint = rule.find( '/' );
    std::string stayAliveIf = rule.substr( 0, gPoint );
    std::string newBornIf = rule.substr( gPoint + 1, strLen - gPoint - 1 );


    for ( int i = 0; i < newBornIf.length( ); ++i )
    {
        std::stringstream ss;
        ss << newBornIf[i];
        int shift;
        ss >> shift;
        ruleMask = ruleMask | 1 << ( shift + 9 );
    }
    for ( int i = 0; i < stayAliveIf.length( ); ++i )
    {
        std::stringstream ss;
        ss << stayAliveIf[i];
        int shift;
        ss >> shift;
        ruleMask = ruleMask | 1 << ( shift );
    }
    std::cout << "newborn if=" << newBornIf << " stay alive if=" << stayAliveIf << " mask=" << ruleMask << std::endl;

    gol::Simulation sim( ruleMask, steps, grid, gpus, endless );
    sim.init( );
    sim.start( );

    MPI_CHECK( MPI_Finalize( ) );
    return 0;
}
