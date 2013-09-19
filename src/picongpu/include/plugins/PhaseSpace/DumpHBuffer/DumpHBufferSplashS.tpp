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
    template<typename Type, uint32_t bufDim>
    void DumpHBuffer::operator()( const PMacc::container::HostBuffer<Type, bufDim>& hBuffer,
                                  const std::pair<uint32_t, uint32_t> axis_element,
                                  const double unit,
                                  const uint32_t currentStep ) const
    {
        const uint32_t maxOpenFilesPerNode = 4;
        DCollector::DomainCollector( maxOpenFilesPerNode ) domainCollector;

        /** Open File *********************************************************/
        std::string fCoords("xyz");

        std::ostringstream filename;
        filename << "phaseSpace/PhaseSpace_"
                 << currentStep;

        try
        {
            domainCollector.open( filename.str().c_str(),
                                  DCollector::DataCollector::FAT_WRITE );
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
        /** \todo type trait -> floatN_X to DCollector Types */
        DCollector::ColTypeDouble ctDouble;
        DCollector::ColTypeFloat  ctFloat;
        
        const PMacc::math::Size_t<bufDim> sizeBuf = hBuffer.size();
        PMacc::SubGrid<simDim>& sg = PMacc::SubGrid<simDim>::getInstance();
        const size_t rOffset = sg.getSimulationBox().getGlobalOffset()[axis_element.first];

        domainCollector.writeDomain( currentStep, ctFloat, bufDim,
                                     DCollector::Dimensions( sizeBuf.x(), sizeBuf.y(), 1 ),
                                     dataSetName.str().c_str(),
                                     DCollector::Dimensions( rOffset, 0, 0 ),
                                     DCollector::Dimensions( sizeBuf.x(), sizeBuf.y(), 1 ),
                                     DomainCollector::GridType,
                                     &(*hBuffer.origin()) );

        /** Write Additional Attributes ***************************************/

        DCollector::Dimensions sim_size( sg.getSimulationBox().getGlobalSize().x(),
                                         sg.getSimulationBox().getGlobalSize().y(),
                                         sg.getSimulationBox().getGlobalSize().z() );
        DCollector::Dimensions sim_global_offset(
                                         sg.getSimulationBox().getGlobalOffset().x(),
                                         sg.getSimulationBox().getGlobalOffset().y(),
                                         sg.getSimulationBox().getGlobalOffset().z() );

        domainCollector.writeAttribute( currentStep,
                                        DCollector::ColTypeDim(),
                                        dataSetName.str().c_str(),
                                        "sim_size",
                                        &sim_size );
        domainCollector.writeAttribute( currentStep,
                                        DCollector::ColTypeDim(),
                                        dataSetName.str().c_str(),
                                        "sim_global_offset",
                                        &sim_global_offset );
        domainCollector.writeAttribute( currentStep,
                                        ctDouble,
                                        dataSetName.str().c_str(),
                                        "sim_unit",
                                        &unit );

        /** Close File ********************************************************/
        domainCollector.close();
    }

} // namespace picongpu
