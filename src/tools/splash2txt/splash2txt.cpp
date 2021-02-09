/* Copyright 2013-2021 Felix Schmitt, Axel Huebl, Rene Widera,
 *                     Alexander Grund
 *
 * This file is part of splash2txt.
 *
 * splash2txt is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * splash2txt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with splash2txt.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "splash2txt.hpp"

#include "tools_splash_parallel.hpp"

#if (ENABLE_ADIOS==1)
#include "tools_adios_parallel.hpp"
#endif

#include <regex>

namespace po = boost::program_options;

const size_t numAllowedSlices = 3;
const char* allowedSlices[numAllowedSlices] = { "xy", "xz", "yz" };
const char* allowedReverseSlices[numAllowedSlices] = { "yx", "zx", "zy" };

std::ostream &errorStream  = std::cerr;

bool parseOptions( int argc, char** argv, ProgramOptions &options )
{
    // add help message
    std::stringstream desc_stream;
    desc_stream << "Usage splash2txt [options] <input-file>" << std::endl;

    po::options_description desc( desc_stream.str( ) );

    std::string slice_string = "";
    std::string filemode = "splash";

#if (ENABLE_ADIOS==1)
    const std::string filemodeOptions = "[splash,adios]";
#else
    const std::string filemodeOptions = "[splash]";
#endif

    // add possible options
    desc.add_options( )
        ( "help,h", "print help message" )
        ( "verbose,v", "verbose output, print status messages" )
        ( "mode,m", po::value< std::string > ( &filemode )->default_value( filemode ), (std::string("File Mode ") + filemodeOptions).c_str() )
        ( "list,l", "list the available datasets for an input file" )
        ( "input-file", po::value< std::string > ( &options.inputFile ), "parallel input file" )
        ( "output-file,o", po::value< std::string > ( &options.outputFile ), "output file (otherwise stdout)" )
        ( "step,s", po::value<uint32_t > ( &options.step )->default_value( options.step ), "requested simulation step" )
        ( "data,d", po::value<std::vector<std::string> > ( &options.data )->multitoken( ), "name of datasets to print" )
        ( "slice", po::value< std::string > ( &slice_string )->default_value( "xy" ), "dimensions of slice for field data, e.g. xy" )
        /// \todo if the standard offset value would be the MIDDLE of the global simulation area, it would be awesome
        ( "offset", po::value<size_t > ( &options.sliceOffset )->default_value( 0 ), "offset of slice in dataset" )
        ( "delimiter", po::value<std::string>( &options.delimiter )->default_value( " " ), "select a delimiter for data elements. default is a single space character" )
        ( "no-units", "no conversion of stored data elements with their respective unit" )
        ;

    po::positional_options_description pos_options;
    pos_options.add( "input-file", -1 );

    try
    {
        // parse command line options and store values in vm
        po::variables_map vm;
        po::store( po::command_line_parser( argc, argv ).
                   options( desc ).positional( pos_options ).run( ), vm );
        po::notify( vm );

        // print help message and quit program if requested or if required parameters are missing
        if ( vm.count( "help" ) || !vm.count( "input-file" ) ||
             ( !vm.count( "data" ) && !vm.count( "list" ) ) ||
             !vm.count( "step" ) )
        {
            errorStream << desc << "\n";
            return false;
        }

        if (filemode == "splash" )
        {
            options.fileMode = FM_SPLASH;
        }
#if (ENABLE_ADIOS==1)
        else if(filemode == "adios")
        {
            options.fileMode = FM_ADIOS;
        }
#endif
        // re-parse wrong typed input files to valid format, if possible
        //   find _X.h5 with syntax at the end and delete it
        std::regex filePattern( "\\(_[[:digit:]]\\)*\\.h5",
                                  std::regex::egrep );
        options.inputFile = std::regex_replace( options.inputFile, filePattern, "" );

        // set various flags
        options.verbose = vm.count( "verbose" ) != 0;
        options.listDatasets = vm.count( "list" ) != 0;
        options.toFile = vm.count( "output-file" ) != 0;
        options.applyUnits = vm.count( "no-units" ) == 0;

        if ( vm.count( "slice" ) ^ vm.count( "offset" ) )
        {
            errorStream << "Parameters 'slice' and 'offset' require each other." << std::endl;
            errorStream << desc << "\n";
            return false;
        }

        if ( vm.count( "slice" ) )
        {
            bool sliceFound = false;
            options.fieldDims.set( 0, 0, 0 );
            for ( size_t i = 0; i < numAllowedSlices; ++i )
            {
                if ( slice_string.compare( allowedSlices[i] ) == 0 )
                {
                    sliceFound = true;
                    options.isReverseSlice = false;
                    break;
                }

                if ( slice_string.compare( allowedReverseSlices[i] ) == 0 )
                {
                    sliceFound = true;
                    options.isReverseSlice = true;
                    break;
                }
            }

            if ( !sliceFound )
            {
                errorStream << "Invalid input for parameter 'slice'. Accepted: xy, xz, yz, yx, zx, zy" << std::endl;
                errorStream << desc << "\n";
                return false;
            }

            if ( slice_string.find( 'x' ) != std::string::npos )
                options.fieldDims[0] = 1;

            if ( slice_string.find( 'y' ) != std::string::npos )
                options.fieldDims[1] = 1;

            if ( slice_string.find( 'z' ) != std::string::npos )
                options.fieldDims[2] = 1;
        }

    }
    catch ( const boost::program_options::error& )
    {
        errorStream << desc << "\n";
        throw std::runtime_error( "Error parsing command line options!" );
    }

    return true;
}

static void mpi_finalize(void)
{
     // PHDF5 might have finalized already
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
}

int main( int argc, char** argv )
{
    int rank, size;
    // we read with one MPI process
    Dims mpi_topology(1, 1, 1);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > 1)
    {
        std::cerr << "Only 1 MPI process supported" << std::endl;
        mpi_finalize();
        return 1;
    }

    // read command line options
    ProgramOptions options;
    bool parseSuccessfull = false;
    std::ostream *outStream = &std::cout;

    options.outputFile = "";
    options.step = 0;

    try
    {
        parseSuccessfull = parseOptions( argc, argv, options );
    }
    catch ( const std::runtime_error& e )
    {
        errorStream << "Error: " << e.what( ) << std::endl;
        mpi_finalize();
        return 1;
    }

    if ( !parseSuccessfull )
    {
        mpi_finalize();
        return 1;
    }

    ITools *tools = nullptr;
    switch ( options.fileMode)
    {
        case FM_SPLASH: tools = new ToolsSplashParallel( options, mpi_topology, *outStream );
                        break;
#if (ENABLE_ADIOS==1)
        case FM_ADIOS: tools = new ToolsAdiosParallel( options, mpi_topology, *outStream );
                        break;
#endif
    }

    try
    {
        std::ofstream file;
        if ( options.toFile )
        {
            file.open( options.outputFile.c_str( ) );
            if ( !file.is_open( ) )
                throw std::runtime_error( "Failed to open output file for writing." );

            outStream = &file;
        }

        // apply requested command to file
        if ( options.listDatasets )
            tools->listAvailableDatasets( );
        else
            tools->convertToText(  );

        if ( options.toFile )
        {
            file.close( );
        }
    }
    catch ( const std::runtime_error& e )
    {
        errorStream << "Error: " << e.what( ) << std::endl;
        delete tools;

        mpi_finalize();
        return 1;
    }

    // cleanup
    delete tools;

    mpi_finalize();
    return 0;
}
