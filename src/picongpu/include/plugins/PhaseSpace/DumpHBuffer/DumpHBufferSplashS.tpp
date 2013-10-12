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

#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "math/vector/Int.hpp"
#include "math/vector/Size_t.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <utility>
#include <iostream>

#include <splash.h>

namespace picongpu
{
    template<typename T_Type, int T_bufDim>
    void DumpHBuffer::operator()( const PMacc::container::HostBuffer<T_Type, T_bufDim>& hBuffer,
                                  const std::pair<uint32_t, uint32_t> axis_element,
                                  const double unit,
                                  const uint32_t currentStep,
                                  MPI_Comm& mpiComm ) const
    {
        typedef T_Type Type;
        const int bufDim = T_bufDim;

        PMacc::GridController<simDim>& gc = PMacc::GridController<simDim>::getInstance();
        PMacc::math::Size_t<simDim> gpuDim = gc.getGpuNodes();
        PMacc::math::Int<simDim> gpuPos = gc.getPosition();

        const uint32_t maxOpenFilesPerNode = 4;
        DCollector::DomainCollector domainCollector( maxOpenFilesPerNode );

        /** Open File *********************************************************/
        std::string fCoords("xyz");

        std::ostringstream filename;
        filename << "phaseSpace/PhaseSpace_"
                 << fCoords.at(axis_element.first) << "_"
                 << currentStep;

        // set attributes for datacollector files
        DCollector::DataCollector::FileCreationAttr attr;
        attr.enableCompression = true;
        attr.fileAccType = DCollector::DataCollector::FAT_WRITE;

        attr.mpiPosition.set(gpuPos[axis_element.first], 0, 0);
        attr.mpiSize.set(gpuDim[axis_element.first], 1, 1);

        try
        {
            domainCollector.open( filename.str().c_str(), attr );
        }
        catch (DCollector::DCException e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("Failed to open DomainCollector");
        }

        /** Dataset Name ******************************************************/
        std::ostringstream dataSetName;
        /* xpx or ypz or ... */
        dataSetName << fCoords.at(axis_element.first)
                    << "p" << fCoords.at(axis_element.second);

        /** Write DataSet *****************************************************/
        DCollector::ColTypeDouble ctDouble;
        typename PICToSplash<Type>::type ctPhaseSpace;

        /** local buffer size (aka splash subdomain) */
        DCollector::Dimensions phaseSpace_size_local( hBuffer.size().x(),
                                                      hBuffer.size().y(),
                                                      1 );

        /** global buffer size (aka splash domain) */
        PMacc::SubGrid<simDim>& sg = PMacc::SubGrid<simDim>::getInstance();
        const size_t rOffset = sg.getSimulationBox().getGlobalOffset()[axis_element.first];
        const size_t rSize = sg.getSimulationBox().getGlobalSize()[axis_element.first];
        DCollector::Dimensions phaseSpace_size( rSize, hBuffer.size().y(), 1 );
        DCollector::Dimensions phaseSpace_global_offset( rOffset, 0, 0 );

        domainCollector.writeDomain( currentStep, ctPhaseSpace, bufDim,
                                     phaseSpace_size_local,
                                     dataSetName.str().c_str(),
                                     phaseSpace_global_offset,
                                     phaseSpace_size_local,
                                     DCollector::DomainCollector::GridType,
                                     &(*hBuffer.origin()) );

        /** Write Additional Attributes ***************************************/
        domainCollector.writeAttribute( currentStep,
                                        DCollector::ColTypeDim(),
                                        dataSetName.str().c_str(),
                                        "sim_size",
                                        phaseSpace_size.getPointer() );
        domainCollector.writeAttribute( currentStep,
                                        DCollector::ColTypeDim(),
                                        dataSetName.str().c_str(),
                                        "sim_global_offset",
                                        phaseSpace_global_offset.getPointer() );
        domainCollector.writeAttribute( currentStep,
                                        ctDouble,
                                        dataSetName.str().c_str(),
                                        "sim_unit",
                                        &unit );

        /** Close File ********************************************************/
        domainCollector.close();
    }

} // namespace picongpu
