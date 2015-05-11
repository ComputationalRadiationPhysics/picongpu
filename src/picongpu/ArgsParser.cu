/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Rene Widera
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

#include <iostream>
#include <sstream>
#include <fstream>

#include "include/ArgsParser.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <cuda.h>
//        cuda_runtime.h
//        cuda_runtime_api.h
#include <mpi.h>
#include <mallocMC/mallocMC.hpp>
#if (ENABLE_HDF5 == 1)
    #include <splash/splash.h>
#endif
#if (ENABLE_ADIOS == 1)
    #include <adios.h>
#endif
#if (ENABLE_PNG == 1)
    #include <pngwriter.h>
#endif
// IceT

namespace picongpu
{
    namespace po = boost::program_options;

    ArgsParser::ArgsParser( )
    {

    }

    ArgsParser::ArgsParser( ArgsParser& )
    {

    }

    template <class T>
    bool from_string( T& t,
                      const std::string& s,
                      std::ios_base& ( *f )( std::ios_base& ) )
    {
        std::istringstream iss( s );
        if ( ( iss >> f >> t ).fail( ) )
            throw std::invalid_argument( "convertion invalid!" );

        return true;
    }

    ArgsParser& ArgsParser::getInstance( )
    {
        static ArgsParser instance;
        return instance;
    }

    bool ArgsParser::parse( int argc, char** argv )
    throw (std::runtime_error )
    {
        try
        {
            // add help message
            std::stringstream desc_stream;
            desc_stream << "Usage picongpu [-d dx=1 dy=1 dz=1] -g width height depth [options]" << std::endl;

            po::options_description desc( desc_stream.str( ) );

            std::vector<std::string> config_files;

            // add possible options
            desc.add_options()
                    ( "help,h", "print help message" )
                    ( "version,v", "print version information" )
                    ( "config,c", po::value<std::vector<std::string> > ( &config_files )->multitoken( ), "Config file(s)" )
                    ;

            // add all options from plugins
            for ( std::list<po::options_description>::iterator iter = options.begin( );
                  iter != options.end( ); ++iter )
                desc.add( *iter );

            // parse command line options and config file and store values in vm
            po::variables_map vm;
            //log<picLog::SIMULATION_STATE > ("parsing command line");
            po::store( boost::program_options::parse_command_line( argc, argv, desc ), vm );

            if ( vm.count( "config" ) )
            {
                std::vector<std::string> conf_files = vm["config"].as<std::vector<std::string> >( );

                for ( std::vector<std::string>::const_iterator iter = conf_files.begin( );
                      iter != conf_files.end( ); ++iter )
                {
                    //log<picLog::SIMULATION_STATE > ("parsing config file '%1%'") % (*iter);
                    std::ifstream config_file_stream( iter->c_str( ) );
                    po::store( boost::program_options::parse_config_file( config_file_stream, desc ), vm );
                }
            }

            po::notify( vm );

            // print help message and quit simulation
            if ( vm.count( "help" ) )
            {
                std::cerr << desc << "\n";
                return false;
            }

            // print version information and quit simulation
            if ( vm.count( "version" ) )
            {
                std::cerr << "PIConGPU:     " << 0 //PICONGPU_VERSION_MAJOR
                          << "."              << 1 //PICONGPU_VERSION_MINOR
                          << "."              << 2 //PICONGPU_VERSION_PATCH
                          << std::endl
                          << "  Build-Type: " << BOOST_PP_STRINGIZE(CMAKE_BUILD_TYPE)
                          << std::endl

                          << std::endl
                          << "Third party:"
                          << std::endl
                          << "  OS:         " << BOOST_PP_STRINGIZE(CMAKE_SYSTEM)
                          << " on "           << BOOST_PP_STRINGIZE(CMAKE_SYSTEM_PROCESSOR)
                          << std::endl
                          << "  CMAKE:      " << BOOST_PP_STRINGIZE(CMAKE_VERSION)
                          << " ("             << BOOST_PP_STRINGIZE(CMAKE_GENERATOR)
                          << ")" << std::endl
                          /** \todo compare runtime CUDA version */
                          << "  CUDA:       " << CUDA_VERSION
                          << std::endl
                          << "  CXX:        " << BOOST_PP_STRINGIZE(CMAKE_CXX_COMPILER_ID)
                          << " ("             << BOOST_PP_STRINGIZE(CMAKE_CXX_COMPILER_VERSION)
                          << ")" << std::endl
                          << "  boost:      " << BOOST_PP_STRINGIZE(BOOST_VERSION)
                          << std::endl
                          << "  MPI:        " << std::endl
                          << "    standard: " << MPI_VERSION
                          << "."              << MPI_SUBVERSION
                          << std::endl
                          << "    flavor:   "
#if defined(OMPI_MAJOR_VERSION)
/* includes derivates such as Bullx MPI, Sun, ... */
                                              << "OpenMPI"
                          << " ("             << OMPI_MAJOR_VERSION
                          << "."              << OMPI_MINOR_VERSION
                          << "."              << OMPI_RELEASE_VERSION
                          << ")"
#elif defined(MPICH_VERSION)
/* includes MPICH2 and MPICH3 and
   derivates such as IBM, Cray, MS, Intel, MVAPICH(2), ... */
                                              << "MPICH"
                          << " ("             << MPICH_VERSION
                          << ")"
#else
                                              << "unknown"
#endif
                          << std::endl
                          << "  mallocMC:   " << MALLOCMC_VERSION_MAJOR
                          << "."              << MALLOCMC_VERSION_MINOR
                          << "."              << MALLOCMC_VERSION_PATCH
                          << std::endl
#if defined(PNGWRITER_VERSION_MAJOR)
                          << "  PNGwriter:  " << PNGWRITER_VERSION_MAJOR
                          << "."              << PNGWRITER_VERSION_MINOR
                          << "."              << PNGWRITER_VERSION_PATCH
                          << std::endl
#elif defined(PNGWRITER_VERSION)
                          /* pre 0.5.5 release */
                          << "  PNGwriter:  " << PNGWRITER_VERSION
                          << std::endl
#endif
#if defined(SPLASH_VERSION_MAJOR)
                          << "  Splash:     " << SPLASH_VERSION_MAJOR
                          << "."              << SPLASH_VERSION_MINOR
                          << "."              << SPLASH_VERSION_PATCH
                          << " (Format "      << SPLASH_FILE_FORMAT_MAJOR
                          << "."              << SPLASH_FILE_FORMAT_MINOR
                          /* future: add HDF5 version splash was build against */
                          << ")" << std::endl
#endif
#if defined(ADIOS_VERSION)
                          << "  ADIOS:      " << BOOST_PP_STRINGIZE(ADIOS_VERSION)
                          << std::endl
#endif
#if defined(ICET_VERSION)
                          << "  IceT:       " << ICET_VERSION
                          << std::endl
#endif
                          ;
                return false;
            }
        }
        catch ( boost::program_options::error& e )
        {
            std::cerr << e.what() << std::endl;
            return false;
        }

        return true;
    }

}
