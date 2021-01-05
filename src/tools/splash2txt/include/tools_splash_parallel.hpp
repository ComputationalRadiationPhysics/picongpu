/* Copyright 2013-2021 Felix Schmitt
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

#ifndef TOOLS_SPLASH_PARALLEL_HPP
#define TOOLS_SPLASH_PARALLEL_HPP

#include <iostream>

#include "splash/splash.h"
#include "ITools.hpp"

using namespace splash;

typedef struct
{
    DataContainer* container;
    double unit;
} ExDataContainer;

class ToolsSplashParallel : public ITools
{
public:

    ToolsSplashParallel(ProgramOptions &options, Dims &mpiTopology, std::ostream &outStream);

    ~ToolsSplashParallel();

    void convertToText();

    void listAvailableDatasets();

protected:
    void printAvailableDatasets(std::vector< DataCollector::DCEntry >& dataTypeNames,
            std::string intentation);

    static bool DCEntryCompare(DataCollector::DCEntry i, DataCollector::DCEntry j);

    void printFields(std::vector<ExDataContainer> fileData);

    void printParticles(std::vector<ExDataContainer> fileData);

    void printElement(DCDataType dataType,
            void* elem, double unit, std::string delimiter);

    ParallelDomainCollector dc;
    std::ostream &errorStream;
};

#endif    /* TOOLS_SPLASH_PARALLEL_HPP */

