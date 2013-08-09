/*
 * Copyright 2013 Felix Schmitt, Axel Huebl, René Widera
 *
 * This file is part of splash2txt. 
 * 
 * splash2txt is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
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

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/regex.hpp>
#include <boost/foreach.hpp>
#include <iomanip>

#include "DomainCollector.hpp"
#include "basetypes/ColTypeDouble.hpp"
#include "basetypes/ColTypeInt.hpp"

namespace po = boost::program_options;
using namespace DCollector;

std::ostream &errorStream = std::cerr;

typedef struct
{
    std::string inputFile; // input file, common part
    std::string outputFile; // output file
    bool toFile; // use output file
    std::string delimiter;
    uint32_t step; // simulation iteration
    std::vector<std::string> data; // names of datasets
    Dimensions fieldDims; // for field data, dimensions of slice, e.g. xy -> (1, 1, 0) 
    size_t sliceOffset; // offset of slice (fieldDims) in dataset
    bool isReverseSlice; // if one of allowedReverseSlices is used
    bool verbose; // verbose output on stdout
    bool listDatasets; // list available datasets
    bool applyUnits; // apply the unit stored in HDF5 to the output data
} ProgramOptions;

typedef struct
{
    DataContainer* container;
    double unit;
} ExDataContainer;

const size_t numAllowedSlices = 3;
const char* allowedSlices[numAllowedSlices] = { "xy", "xz", "yz" };
const char* allowedReverseSlices[numAllowedSlices] = { "yx", "zx", "zy" };

bool parseOptions( int argc, char** argv, ProgramOptions &options )
throw (std::runtime_error )
{
    try
    {
        // add help message
        std::stringstream desc_stream;
        desc_stream << "Usage splash2txt [options] <input-file>" << std::endl;

        po::options_description desc( desc_stream.str( ) );

        std::string slice_string = "";

        // add possible options
        desc.add_options( )
            ( "help,h", "print help message" )
            ( "verbose,v", "verbose output, print status messages" )
            ( "list,l", "list the available datasets for an input file and quit" )
            ( "input-file", po::value< std::string > ( &options.inputFile ), "input file" )
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

        // re-parse wrong typed input files to valid format, if possible
        //   find _X_Y_Z.h5 with syntax at the end and delete it
        boost::regex filePattern( "_.*_.*_.*\\.h5",
                                  boost::regex_constants::icase |
                                  boost::regex_constants::perl );
        options.inputFile = boost::regex_replace( options.inputFile, filePattern, "" );

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
            for ( int i = 0; i < numAllowedSlices; ++i )
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
    catch ( boost::program_options::error e )
    {
        throw std::runtime_error( "Error parsing command line options!" );
    }

    return true;
}

void printElement( std::ostream& outStream, DCDataSet::DCDataType dataType,
                   void* elem, double unit, std::string delimiter )
{
    std::stringstream stream;

    switch ( dataType )
    {
    case DCDataSet::DCDT_FLOAT32:
        stream << std::setprecision( 16 ) << *( (float*) elem ) *
            unit << delimiter;
        break;
    case DCDataSet::DCDT_FLOAT64:
        stream << std::setprecision( 16 ) << *( (double*) elem ) *
            unit << delimiter;
        break;
    case DCDataSet::DCDT_UINT32:
        stream << *( (uint32_t*) elem ) * unit << delimiter;
        break;
    case DCDataSet::DCDT_UINT64:
        stream << *( (uint64_t*) elem ) * unit << delimiter;
        break;
    case DCDataSet::DCDT_INT32:
        stream << *( (int32_t*) elem ) * unit << delimiter;
        break;
    case DCDataSet::DCDT_INT64:
        stream << *( (int64_t*) elem ) * unit << delimiter;
        break;
    default:
        throw DCException( "cannot identify datatype" );
    }

    outStream << stream.str( );
}

void printParticles( ProgramOptions &options,
                     std::vector<ExDataContainer> fileData,
                     DomainCollector &dc, std::ostream &outStream )
{
    if ( fileData.size( ) > 0 )
    {
        size_t num_elements = fileData[0].container->getNumElements( );

        if ( options.verbose )
        {
            errorStream << "num_elements = " << num_elements << std::endl;
            errorStream << "container = " << fileData.size( ) << std::endl;
        }

        std::map<DCollector::DataContainer*, DCollector::DomainData*> subdomain;
        std::map<DCollector::DataContainer*, size_t> subdomainIndex;
        std::map<DCollector::DataContainer*, size_t> numElementsProcessed;
        for ( size_t i = 0; i < num_elements; ++i )
        {
            for ( std::vector<ExDataContainer>::iterator iter = fileData.begin( );
                  iter != fileData.end( ); ++iter )
            {
                DCollector::DataContainer *container = iter->container;
                DCDataSet::DCDataType data_type =
                    container->getIndex( 0 )->getDataType( );

                if ( i == 0 )
                {
                    if ( options.verbose )
                        errorStream << "container " << container << " has " <<
                        container->getNumSubdomains( ) << " subdomains" << std::endl;
                    subdomainIndex[container] = 0;
                    subdomain[container] = container->getIndex( subdomainIndex[container] );
                    numElementsProcessed[container] = 0;
                    if ( options.verbose )
                        errorStream << "Loading domaindata 0 for container " << container
                        << " (element = " << i << ")" << std::endl;
                    dc.readDomainLazy( subdomain[container] );
                }
                else
                {
                    if ( numElementsProcessed[container] == subdomain[container]->getElements( ).getDimSize( ) )
                    {
                        subdomain[container]->freeData( );
                        subdomainIndex[container]++;
                        numElementsProcessed[container] = 0;
                        subdomain[container] = container->getIndex( subdomainIndex[container] );
                        if ( options.verbose )
                            errorStream << std::endl << "Loading domaindata " << subdomainIndex[container] <<
                            " for container " << container << " (element = " << i << ")" << std::endl;
                        dc.readDomainLazy( subdomain[container] );
                    }
                }

                void* element = container->getElement( i );
                assert( element != NULL );

                printElement( outStream, data_type, element, iter->unit, options.delimiter );
                numElementsProcessed[container]++;
            }

            outStream << std::endl;

            if ( options.verbose && i % 100000 == 0 )
                errorStream << "." << std::flush;
        }

        if ( options.verbose )
            errorStream << std::endl;
    }
}

void printFields( ProgramOptions &options,
                  std::vector<ExDataContainer> fileData, std::ostream &outStream )
{
    if ( fileData.size( ) > 0 )
    {
        if ( options.verbose )
            errorStream << "container = " << fileData.size( ) << std::endl;

        Dimensions domain_size = fileData[0].container->getSize( );
        size_t size1, size2;
        if ( options.fieldDims[0] )
        {
            size1 = domain_size[0];
            if ( options.fieldDims[1] )
                size2 = domain_size[1];
            else
                size2 = domain_size[2];
        }
        else
        {
            size1 = domain_size[1];
            size2 = domain_size[2];
        }

        if ( options.isReverseSlice )
        {
            size_t tmpSize = size1;
            size1 = size2;
            size2 = tmpSize;
        }

        for ( size_t j = 0; j < size2; ++j )
        {
            size_t index = 0;

            for ( size_t i = 0; i < size1; ++i )
            {
                if ( !options.isReverseSlice )
                    index = j * size1 + i;
                else
                    index = i * size2 + j;

                for ( std::vector<ExDataContainer>::iterator iter = fileData.begin( );
                      iter != fileData.end( ); ++iter )
                {
                    DCDataSet::DCDataType data_type =
                        iter->container->getIndex( 0 )->getDataType( );

                    void* element = iter->container->getElement( index );
                    assert( element != NULL );

                    printElement( outStream, data_type, element, iter->unit, options.delimiter );
                }
            }

            outStream << std::endl;

            if ( options.verbose && index % 100000 == 0 )
                errorStream << "." << std::flush;
        }

        if ( options.verbose )
            errorStream << std::endl;
    }
}

void convertToText( DomainCollector &dc, ProgramOptions &options, std::ostream &outStream )
{
    if ( options.data.size( ) == 0 )
        throw std::runtime_error( "No datasets requested" );

    ColTypeInt ctInt;
    ColTypeDouble ctDouble;

    // read data
    //

    std::vector<ExDataContainer> file_data;

    // identify reference data class (poly, grid), reference domain size and number of elements
    //

    DomainCollector::DomDataClass ref_data_class = DomainCollector::UndefinedType;
    dc.readAttribute( options.step, options.data[0].c_str( ), DOMCOL_ATTR_CLASS,
                      &ref_data_class, NULL );

    switch ( ref_data_class )
    {
    case DomainCollector::GridType:
        if ( options.verbose )
            errorStream << "Converting GRID data" << std::endl;
        break;
    case DomainCollector::PolyType:
        if ( options.verbose )
            errorStream << "Converting POLY data" << std::endl;
        break;
    default:
        throw std::runtime_error( "Could not identify data class for requested dataset" );
    }

    Domain ref_total_domain;
    ref_total_domain = dc.getTotalDomain( options.step, options.data[0].c_str( ) );

    size_t num_elements = dc.getTotalElements( options.step, options.data[0].c_str( ) );

    for ( std::vector<std::string>::const_iterator iter = options.data.begin( );
          iter != options.data.end( ); ++iter )
    {
        // check that all datasets match to each other
        //

        DomainCollector::DomDataClass data_class = DomainCollector::UndefinedType;
        dc.readAttribute( options.step, iter->c_str( ), DOMCOL_ATTR_CLASS,
                          &data_class, NULL );
        if ( data_class != ref_data_class )
            throw std::runtime_error( "All requested datasets must be of the same data class" );

        if ( dc.getTotalElements( options.step, iter->c_str( ) ) != num_elements )
            throw std::runtime_error( "All requested datasets must contain same number of elements" );

        Domain total_domain = dc.getTotalDomain( options.step, iter->c_str( ) );
        if ( total_domain != ref_total_domain )
            throw std::runtime_error( "All requested datasets must map to the same domain" );

        // create an extended container for each dataset
        ExDataContainer excontainer;

        if ( ref_data_class == DomainCollector::PolyType )
        {
            // poly type

            excontainer.container = dc.readDomain( options.step, iter->c_str( ),
                                                   total_domain.getStart( ), total_domain.getSize( ), NULL, true );
        }
        else
        {
            // grid type

            Dimensions offset( total_domain.getStart( ) );
            Dimensions domain_size( total_domain.getSize( ) );

            for ( int i = 0; i < 3; ++i )
                if ( options.fieldDims[i] == 0 )
                {
                    offset[i] = options.sliceOffset;
                    domain_size[i] = 1;
                    break;
                }

            excontainer.container = dc.readDomain( options.step, iter->c_str( ),
                                                   offset, domain_size, NULL );
        }

        // read unit
        //
        if ( options.applyUnits )
        {
            try
            {
                dc.readAttribute( options.step, iter->c_str( ), "sim_unit",
                                  &( excontainer.unit ), NULL );
            }
            catch ( DCException e )
            {
                if ( options.verbose )
                    errorStream << "no unit for '" << iter->c_str( ) << "', defaulting to 1.0" << std::endl;
                excontainer.unit = 1.0;
            }

            if ( options.verbose )
                errorStream << "Loaded dataset '" << iter->c_str( ) << "' with unit '" <<
                excontainer.unit << "'" << std::endl;
        }
        else
        {
            excontainer.unit = 1.0;
            if ( options.verbose )
                errorStream << "Loaded dataset '" << iter->c_str( ) << "'" << std::endl;
        }

        file_data.push_back( excontainer );
    }

    assert( file_data[0].container->get( 0 )->getData( ) != NULL );

    // write to file
    //
    if ( ref_data_class == DomainCollector::PolyType )
        printParticles( options, file_data, dc, outStream );
    else
        printFields( options, file_data, outStream );

    for ( std::vector<ExDataContainer>::iterator iter = file_data.begin( );
          iter != file_data.end( ); ++iter )
    {
        delete iter->container;
    }
}

bool DCEntryCompare( DataCollector::DCEntry i, DataCollector::DCEntry j )
{
    return (i.name < j.name );
}

void printAvailableDatasets( std::vector< DataCollector::DCEntry >& dataTypeNames,
                             std::string intentation, std::ostream &outStream )
{
    std::sort( dataTypeNames.begin( ), dataTypeNames.end( ), DCEntryCompare );

    std::string lastdataName = "";
    size_t matchingLength, lastMatchingUnderscore;

    BOOST_FOREACH( DataCollector::DCEntry dataName, dataTypeNames )
    {
        matchingLength = 0;
        lastMatchingUnderscore = 0;

        while ( dataName.name.compare( matchingLength, 1, lastdataName, matchingLength, 1 ) == 0 )
        {
            if ( dataName.name[matchingLength] == '_' )
                lastMatchingUnderscore = matchingLength;
            matchingLength++;
        }

        // coordinates at the end
        if ( matchingLength == dataName.name.size( ) - 1 )
            outStream << '/'
            << dataName.name.substr( matchingLength );
            // new parameters which differ in more than the coordinate at the end
        else
        {
            outStream << std::string( matchingLength == 0, '\n' ) << '\n'
                << intentation
                << std::string( lastMatchingUnderscore, ' ' )
                << dataName.name.substr( lastMatchingUnderscore );
        }

        lastdataName = dataName.name;
    }
    outStream << std::endl;
}

void listAvailableDatasets( DomainCollector& dc, std::ostream &outStream )
{
    // number of timesteps in this file
    size_t num_entries = 0;
    dc.getEntryIDs( NULL, &num_entries );
    if ( num_entries == 0 )
    {
        outStream << "no entries in file" << std::endl;
        return;
    }

    int32_t *entries = new int32_t[num_entries];
    dc.getEntryIDs( entries, NULL );
    std::sort( entries, entries + num_entries );

    outStream << std::endl
        << "first dump at: " << entries[0] << std::endl
        << "number of dumps: " << num_entries << std::endl;
    if ( num_entries > 1 )
        outStream << "spacing between dumps 0-1: "
        << entries[1] - entries[0] << std::endl
        << std::endl;

    outStream << "available time steps:";
    for ( int i = 0; i < num_entries; ++i )
        outStream << " " << entries[i];
    outStream << std::endl;

    // available data sets in this file
    std::vector<DataCollector::DCEntry> dataTypeNames;
    size_t numDataTypes = 0;
    dc.getEntriesForID( entries[0], NULL, &numDataTypes );
    dataTypeNames.resize( numDataTypes );
    dc.getEntriesForID( entries[0], &( dataTypeNames.front( ) ), NULL );

    // parse dataTypeNames vector in a nice way for stdout
    outStream << "Available data field names:";
    printAvailableDatasets( dataTypeNames, "  ", outStream );

    // global cell size and start
    Domain totalDomain = dc.getTotalDomain( entries[0], ( dataTypeNames.front( ).name.c_str( ) ) );
    outStream << std::endl
        << "Total domain: "
        << totalDomain.toString( ) << std::endl;

    delete[] entries;
}

int main( int argc, char** argv )
{
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
    catch ( std::runtime_error e )
    {
        errorStream << "Error: " << e.what( ) << std::endl;
        return 1;
    }

    if ( !parseSuccessfull )
        return 1;

    const uint32_t maxOpenFilesPerNode = 100;
    DomainCollector dc( maxOpenFilesPerNode );

    DataCollector::FileCreationAttr fattr;
    fattr.enableCompression = false;
    fattr.fileAccType = DataCollector::FAT_READ_MERGED;
    fattr.mpiPosition.set( 0, 0, 0 );
    fattr.mpiSize.set( 0, 0, 0 );

    if ( options.verbose )
        errorStream << options.inputFile << std::endl;

    dc.open( options.inputFile.c_str( ), fattr );

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

        if ( options.listDatasets )
            listAvailableDatasets( dc, *outStream );
        else
            convertToText( dc, options, *outStream );

        if ( options.toFile )
        {
            file.close( );
        }
    }
    catch ( std::runtime_error e )
    {
        errorStream << "Error: " << e.what( ) << std::endl;
        dc.close( );
        return 1;
    }

    dc.close( );

    return 0;
}
