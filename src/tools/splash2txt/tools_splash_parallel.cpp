/*
 * Copyright 2013-2014 Felix Schmitt, Axel Huebl, Rene Widera
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

#include <boost/foreach.hpp>
#include <algorithm>

#include "tools_splash_parallel.hpp"

ToolsSplashParallel::ToolsSplashParallel(ProgramOptions &options, Dims &mpiTopology, std::ostream &outStream) :
ITools(options, mpiTopology, outStream),
dc(MPI_COMM_WORLD, MPI_INFO_NULL, Dimensions(mpiTopology[0], mpiTopology[1], mpiTopology[2]), 100),
errorStream(std::cerr)
{
    DataCollector::FileCreationAttr fattr;
    fattr.enableCompression = false;
    fattr.fileAccType = DataCollector::FAT_READ_MERGED;
    fattr.mpiPosition.set(0, 0, 0);
    fattr.mpiSize.set(0, 0, 0);

    if (options.verbose)
        errorStream << options.inputFile << std::endl;

    dc.open(options.inputFile.c_str(), fattr);
}

ToolsSplashParallel::~ToolsSplashParallel()
{
    dc.close();
    dc.finalize();
}

void ToolsSplashParallel::printElement(DCDataType dataType,
        void* elem, double unit, std::string delimiter)
{
    std::stringstream stream;

    switch (dataType)
    {
        case DCDT_FLOAT32:
            stream << std::setprecision(16) << *((float*) elem) *
                    unit << delimiter;
            break;
        case DCDT_FLOAT64:
            stream << std::setprecision(16) << *((double*) elem) *
                    unit << delimiter;
            break;
        case DCDT_UINT32:
            stream << *((uint32_t*) elem) * unit << delimiter;
            break;
        case DCDT_UINT64:
            stream << *((uint64_t*) elem) * unit << delimiter;
            break;
        case DCDT_INT32:
            stream << *((int32_t*) elem) * unit << delimiter;
            break;
        case DCDT_INT64:
            stream << *((int64_t*) elem) * unit << delimiter;
            break;
        default:
            throw DCException("cannot identify datatype");
    }

    outStream << stream.str();
}

void ToolsSplashParallel::printParticles(std::vector<ExDataContainer> fileData)
{
    if (fileData.size() > 0)
    {
        size_t num_elements = fileData[0].container->getNumElements();

        if (options.verbose)
        {
            errorStream << "num_elements = " << num_elements << std::endl;
            errorStream << "container = " << fileData.size() << std::endl;
        }

        std::map<DataContainer*, DomainData*> subdomain;
        std::map<DataContainer*, size_t> subdomainIndex;
        std::map<DataContainer*, size_t> numElementsProcessed;
        for (size_t i = 0; i < num_elements; ++i)
        {
            for (std::vector<ExDataContainer>::iterator iter = fileData.begin();
                    iter != fileData.end(); ++iter)
            {
                DataContainer *container = iter->container;
                DCDataType data_type =
                        container->getIndex(0)->getDataType();

                if (i == 0)
                {
                    if (options.verbose)
                        errorStream << "container " << container << " has " <<
                            container->getNumSubdomains() << " subdomains" << std::endl;
                    subdomainIndex[container] = 0;
                    subdomain[container] = container->getIndex(subdomainIndex[container]);
                    numElementsProcessed[container] = 0;
                    if (options.verbose)
                        errorStream << "Loading domaindata 0 for container " << container
                            << " (element = " << i << ")" << std::endl;
                    dc.readDomainLazy(subdomain[container]);
                } else
                {
                    if (numElementsProcessed[container] == subdomain[container]->getElements().getScalarSize())
                    {
                        subdomain[container]->freeData();
                        subdomainIndex[container]++;
                        numElementsProcessed[container] = 0;
                        subdomain[container] = container->getIndex(subdomainIndex[container]);
                        if (options.verbose)
                            errorStream << std::endl << "Loading domaindata " << subdomainIndex[container] <<
                                " for container " << container << " (element = " << i << ")" << std::endl;
                        dc.readDomainLazy(subdomain[container]);
                    }
                }

                void* element = container->getElement(i);
                assert(element != NULL);

                printElement(data_type, element, iter->unit, options.delimiter);
                numElementsProcessed[container]++;
            }

            outStream << std::endl;

            if (options.verbose && i % 100000 == 0)
                errorStream << "." << std::flush;
        }

        if (options.verbose)
            errorStream << std::endl;
    }
}

void ToolsSplashParallel::printFields(std::vector<ExDataContainer> fileData)
{
    if (fileData.size() > 0)
    {
        if (options.verbose)
            errorStream << "container = " << fileData.size() << std::endl;

        Dimensions domain_size = fileData[0].container->getSize();
        size_t size1, size2;
        if (options.fieldDims[0])
        {
            size1 = domain_size[0];
            if (options.fieldDims[1])
                size2 = domain_size[1];
            else
                size2 = domain_size[2];
        } else
        {
            size1 = domain_size[1];
            size2 = domain_size[2];
        }

        if (options.isReverseSlice)
        {
            size_t tmpSize = size1;
            size1 = size2;
            size2 = tmpSize;
        }

        for (size_t j = 0; j < size2; ++j)
        {
            size_t index = 0;

            for (size_t i = 0; i < size1; ++i)
            {
                if (!options.isReverseSlice)
                    index = j * size1 + i;
                else
                    index = i * size2 + j;

                for (std::vector<ExDataContainer>::iterator iter = fileData.begin();
                        iter != fileData.end(); ++iter)
                {
                    DCDataType data_type =
                            iter->container->getIndex(0)->getDataType();

                    void* element = iter->container->getElement(index);
                    assert(element != NULL);

                    printElement(data_type, element, iter->unit, options.delimiter);
                }
            }

            outStream << std::endl;

            if (options.verbose && index % 100000 == 0)
                errorStream << "." << std::flush;
        }

        if (options.verbose)
            errorStream << std::endl;
    }
}

void ToolsSplashParallel::convertToText()
{
    if (options.data.size() == 0)
        throw std::runtime_error("No datasets requested");

    ColTypeInt ctInt;
    ColTypeDouble ctDouble;

    // read data
    //

    std::vector<ExDataContainer> file_data;

    // identify reference data class (poly, grid), reference domain size and number of elements
    //

    DomainCollector::DomDataClass ref_data_class = DomainCollector::UndefinedType;
    try
    {
        dc.readAttribute(options.step, options.data[0].c_str(), DOMCOL_ATTR_CLASS,
                &ref_data_class, NULL);
    } catch (DCException)
    {
        errorStream << "Error: No domain information for dataset '" << options.data[0] << "' available." << std::endl;
        errorStream << "This might not be a valid libSplash domain." << std::endl;
        return;
    }

    switch (ref_data_class)
    {
        case DomainCollector::GridType:
            if (options.verbose)
                errorStream << "Converting GRID data" << std::endl;
            break;
        case DomainCollector::PolyType:
            if (options.verbose)
                errorStream << "Converting POLY data" << std::endl;
            break;
        default:
            throw std::runtime_error("Could not identify data class for requested dataset");
    }

    Domain ref_total_domain;
    ref_total_domain = dc.getGlobalDomain(options.step, options.data[0].c_str());

    for (std::vector<std::string>::const_iterator iter = options.data.begin();
            iter != options.data.end(); ++iter)
    {
        // check that all datasets match to each other
        //

        DomainCollector::DomDataClass data_class = DomainCollector::UndefinedType;
        dc.readAttribute(options.step, iter->c_str(), DOMCOL_ATTR_CLASS,
                &data_class, NULL);
        if (data_class != ref_data_class)
            throw std::runtime_error("All requested datasets must be of the same data class");

        Domain total_domain = dc.getGlobalDomain(options.step, iter->c_str());
        if (total_domain != ref_total_domain)
            throw std::runtime_error("All requested datasets must map to the same domain");

        // create an extended container for each dataset
        ExDataContainer excontainer;

        if (ref_data_class == DomainCollector::PolyType)
        {
            // poly type

            excontainer.container = dc.readDomain(options.step, iter->c_str(),
                    Domain(total_domain.getOffset(), total_domain.getSize()), NULL, true);
        } else
        {
            // grid type

            Dimensions offset(total_domain.getOffset());
            Dimensions domain_size(total_domain.getSize());

            for (int i = 0; i < 3; ++i)
                if (options.fieldDims[i] == 0)
                {
                    offset[i] = options.sliceOffset;
                    if (offset[i] > total_domain.getBack()[i])
                        throw DCException("Requested offset outside of domain");

                    domain_size[i] = 1;
                    break;
                }

            excontainer.container = dc.readDomain(options.step, iter->c_str(),
                    Domain(offset, domain_size), NULL);
        }

        // read unit
        //
        if (options.applyUnits)
        {
            try
            {
                dc.readAttribute(options.step, iter->c_str(), "sim_unit",
                        &(excontainer.unit), NULL);
            } catch (DCException e)
            {
                if (options.verbose)
                    errorStream << "no unit for '" << iter->c_str() << "', defaulting to 1.0" << std::endl;
                excontainer.unit = 1.0;
            }

            if (options.verbose)
                errorStream << "Loaded dataset '" << iter->c_str() << "' with unit '" <<
                excontainer.unit << "'" << std::endl;
        } else
        {
            excontainer.unit = 1.0;
            if (options.verbose)
                errorStream << "Loaded dataset '" << iter->c_str() << "'" << std::endl;
        }

        file_data.push_back(excontainer);
    }

    assert(file_data[0].container->get(0)->getData() != NULL);

    // write to file
    //
    if (ref_data_class == DomainCollector::PolyType)
        printParticles(file_data);
    else
        printFields(file_data);

    for (std::vector<ExDataContainer>::iterator iter = file_data.begin();
            iter != file_data.end(); ++iter)
    {
        delete iter->container;
    }
}

bool ToolsSplashParallel::DCEntryCompare(DataCollector::DCEntry i, DataCollector::DCEntry j)
{
    return (i.name < j.name);
}

void ToolsSplashParallel::printAvailableDatasets(std::vector< DataCollector::DCEntry >& dataTypeNames,
        std::string intentation)
{
    std::sort(dataTypeNames.begin(), dataTypeNames.end(), DCEntryCompare);

    std::string lastdataName = "";
    size_t matchingLength, lastMatchingDelimiter;

    BOOST_FOREACH(DataCollector::DCEntry dataName, dataTypeNames)
    {
        matchingLength = 0;
        lastMatchingDelimiter = 0;

        while ( ( dataName.name.size() >= matchingLength ) &&
                ( dataName.name.compare(matchingLength, 1, lastdataName, matchingLength, 1) == 0 ) )
        {
            if (dataName.name[matchingLength] == '/')
                lastMatchingDelimiter = matchingLength;
            matchingLength++;
        }

        // additional linebreak for new top-level group
        outStream << std::string(matchingLength == 0, '\n');
        // align new entry with last matching group
        outStream << intentation << std::string(lastMatchingDelimiter, ' ')
                  << dataName.name.substr(lastMatchingDelimiter)
                  << std::endl;

        lastdataName = dataName.name;
    }
    outStream << std::endl;
}

void ToolsSplashParallel::listAvailableDatasets()
{
    // number of timesteps in this file
    size_t num_entries = 0;
    dc.getEntryIDs(NULL, &num_entries);
    if (num_entries == 0)
    {
        outStream << "no entries in file" << std::endl;
        return;
    }

    int32_t *entries = new int32_t[num_entries];
    dc.getEntryIDs(entries, NULL);
    std::sort(entries, entries + num_entries);

    outStream << std::endl
            << "first dump at: " << entries[0] << std::endl
            << "number of dumps: " << num_entries << std::endl;
    if (num_entries > 1)
        outStream << "spacing between dumps 0-1: "
            << entries[1] - entries[0] << std::endl
            << std::endl;

    outStream << "available time steps:";
    for (size_t i = 0; i < num_entries; ++i)
        outStream << " " << entries[i];
    outStream << std::endl;

    // available data sets in this file
    std::vector<DataCollector::DCEntry> dataTypeNames;
    size_t numDataTypes = 0;
    dc.getEntriesForID(entries[0], NULL, &numDataTypes);
    dataTypeNames.resize(numDataTypes);
    dc.getEntriesForID(entries[0], &(dataTypeNames.front()), NULL);

    // parse dataTypeNames vector in a nice way for stdout
    outStream << "Available data field names:";
    printAvailableDatasets(dataTypeNames, "  ");

    // global cell size and start
    try
    {
        Domain totalDomain = dc.getGlobalDomain(entries[0], (dataTypeNames.front().name.c_str()));
        outStream << std::endl
                << "Global domain: "
                << totalDomain.toString() << std::endl;
    } catch (DCException)
    {
        outStream << std::endl << "(No domain information)" << std::endl;
    }

    delete[] entries;
}
