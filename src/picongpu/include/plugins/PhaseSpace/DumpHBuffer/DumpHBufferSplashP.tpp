/**
 * Copyright 2013 Axel Huebl
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

#pragma once

#include "mpi.h"
#include <splash.h>

#include "communication/manager_common.h"
#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "dimensions/DataSpace.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "math/vector/Int.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <utility>

namespace picongpu
{
    template<typename Type, int bufDim>
    void DumpHBuffer::operator()( const PMacc::container::HostBuffer<Type, bufDim>& hBuffer,
                                  const std::pair<uint32_t, uint32_t> axis_element,
                                  const double unit,
                                  const uint32_t currentStep,
                                  MPI_Comm& mpiComm ) const
    {
        using namespace DCollector;

        std::ostringstream filename;
        filename << "phaseSpace/PhaseSpace_"
                 << currentStep;

        /** get my rank in the fileWriter communicator ************************/
        int size, rank;
        MPI_CHECK(MPI_Comm_size( mpiComm, &size ));
        MPI_CHECK(MPI_Comm_rank( mpiComm, &rank ));

        /** create parallel domain collector **********************************/
        ParallelDomainCollector pdc(
            mpiComm, MPI_INFO_NULL, Dimensions(size, 1, 1), 10 );

        DataCollector::FileCreationAttr fAttr;
        DataCollector::initFileCreationAttr(fAttr);
        fAttr.fileAccType = DataCollector::FAT_CREATE;

        pdc.open( filename.str().c_str(), fAttr );

        /** calculate global size of the phase space **************************/
        PMacc::SubGrid<simDim>& sg = PMacc::SubGrid<simDim>::getInstance();
        const size_t rOffset = sg.getSimulationBox().getGlobalOffset()[axis_element.first];
        const size_t rSize = sg.getSimulationBox().getGlobalSize()[axis_element.first];
        DCollector::Dimensions phaseSpace_size( rSize, hBuffer.size().y(), 1 );
        DCollector::Dimensions phaseSpace_global_offset( rOffset, 0, 0 );

        /** local buffer size (aka splash subdomain) **************************/
        DCollector::Dimensions phaseSpace_size_local( hBuffer.size().x(),
                                                      hBuffer.size().y(),
                                                      1 );

        /** Dataset Name ******************************************************/
        std::string fCoords("xyz");
        std::ostringstream dataSetName;
        /* xpx or ypz or ... */
        dataSetName << fCoords.at(axis_element.first)
                    << "p" << fCoords.at(axis_element.second);

        /** reserve global domain *********************************************/
        ColTypeFloat ctFloat;
        pdc.reserveDomain( currentStep, phaseSpace_size, rank, ctFloat,
                           dataSetName.str().c_str(), Dimensions(0, 0, 0),
                           phaseSpace_size, IDomainCollector::GridType );

        /** write local domain ************************************************/
        pdc.append( currentStep, phaseSpace_size_local, rank,
                    phaseSpace_global_offset, dataSetName.str().c_str(),
                    &(*hBuffer.origin()) );

        ColTypeDouble ctDouble;
        pdc.writeAttribute( currentStep, ctDouble, dataSetName.str().c_str(),
                            "sim_unit", &unit );

        /** close file ********************************************************/
        pdc.close();
    }

} // namespace picongpu
