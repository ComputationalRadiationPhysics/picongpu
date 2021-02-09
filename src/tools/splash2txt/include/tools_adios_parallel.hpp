/*
 *Copyright 2014-2021 Felix Schmitt, Conrad Schumann
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

#ifndef TOOLS_ADIOS_PARALLEL_HPP
#define TOOLS_ADIOS_PARALLEL_HPP

#include <iostream>

#include <adios.h>
#include <adios_read.h>
#include "ITools.hpp"

class ToolsAdiosParallel : public ITools
{
private:

    MPI_Comm comm;
    ADIOS_FILE *pFile;

public:

    ToolsAdiosParallel(ProgramOptions &options, Dims &mpiTopology, std::ostream &outStream);

    ~ToolsAdiosParallel();

    void convertToText();

    void printValue(ADIOS_DATATYPES ptype, void* pValue);

    void listAvailableDatasets();

protected:

    std::ostream &errorStream;

};
#endif
