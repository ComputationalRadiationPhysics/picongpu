/* Copyright 2014-2021 Felix Schmitt, Conrad Schumann, Axel Huebl
 *
 * This file is part of splash2txt.
 *
 * splash2txt is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
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

#include <algorithm>
#include <cstdlib>

#include "tools_adios_parallel.hpp"


ToolsAdiosParallel::ToolsAdiosParallel(ProgramOptions &options, Dims &mpiTopology, std::ostream &outStream) :
ITools(options, mpiTopology, outStream), errorStream(std::cerr)
{
    if (m_options.verbose)
        errorStream << m_options.inputFile << std::endl;

    comm = MPI_COMM_WORLD;
    pFile = adios_read_open_file(m_options.inputFile.c_str(), ADIOS_READ_METHOD_BP, comm);
}

ToolsAdiosParallel::~ToolsAdiosParallel()
{
    adios_read_close(pFile);
    adios_read_finalize_method(ADIOS_READ_METHOD_BP);
}

void ToolsAdiosParallel::convertToText()
{
    if(m_options.data.size() == 0)
        throw std::runtime_error("No datasets requested");

    for (size_t i = 0; i < m_options.data.size(); ++i)
    {
        ADIOS_VARINFO *pVarInfo;

        //get name of dataset to print
        std::string nodeName = m_options.data[i];

        uint8_t *P;
        int varElement = 1;
        int varTypeSize = 0;

        adios_read_init_method(ADIOS_READ_METHOD_BP, comm, nodeName.c_str());

        pVarInfo = adios_inq_var(pFile, nodeName.c_str());

        varTypeSize = adios_type_size(pVarInfo->type, nullptr);

        // get number of elements combined in a dataset
        for(int j = 0; j < pVarInfo->ndim; j++)
        {
            varElement = varElement * pVarInfo->dims[j];
        }
        // allocate memory
        P = (uint8_t*) malloc (sizeof(uint8_t) * varTypeSize * varElement);

        adios_schedule_read(pFile, nullptr, nodeName.c_str(), 0, 1, P);

        adios_perform_reads(pFile, 1);

        if(pVarInfo->ndim > 0)
        {
            for(int k = 0; k < varElement; k++)
            {
                printValue(pVarInfo->type, &P[k*varTypeSize]);
            }
        }
        else
        {
            printValue(pVarInfo->type, pVarInfo->value);
        }
        adios_free_varinfo(pVarInfo);
    }
}

void ToolsAdiosParallel::printValue(ADIOS_DATATYPES pType, void* pValue)
{
     switch(pType)
     {
         case adios_real:
             m_outStream << std::setprecision(16) << *((float*)pValue) << std::endl;
             break;
         case adios_double:
             m_outStream << std::setprecision(16) << *((double*)pValue) << std::endl;
             break;
         case adios_long_double:
             m_outStream << std::setprecision(16) << *((long double*)pValue) << std::endl;
             break;
         case adios_integer:
             m_outStream << *((int32_t*)pValue) << std::endl;
             break;
         case adios_unsigned_integer:
             m_outStream << *((uint32_t*)pValue) << std::endl;
             break;
         case adios_long:
             m_outStream << *((int64_t*)pValue) << std::endl;
             break;
         case adios_unsigned_long:
             m_outStream << *((uint64_t*)pValue) << std::endl;
             break;
         default:
             if (m_options.verbose)
                 errorStream << "unknown data type" << std::endl;
             break;
     }
}

void ToolsAdiosParallel::listAvailableDatasets()
{
    ADIOS_VARINFO *pVarInfo;

    //available data sets in this file
    m_outStream << "Number of available data sets: ";
    m_outStream << pFile->nvars << std::endl;

    for(int i = 0; i < pFile->nvars; i++)
    {
        m_outStream << pFile->var_namelist[i] << " ";

        pVarInfo = adios_inq_var(pFile, pFile->var_namelist[i]);

        if(pVarInfo->ndim > 0)
        {
            // print number of elements per dimension to m_outstream
            m_outStream << "[";
            for(int j = 0; j < pVarInfo->ndim; j++)
            {
                m_outStream << pVarInfo->dims[j];
                if(j < pVarInfo->ndim-1)
                {
                    m_outStream << ",";
                }
            }
            m_outStream << "]" << std::endl;
        }
        else
        {
            m_outStream << "[scalar]" << std::endl;
        }
        adios_free_varinfo(pVarInfo);
    }
}
