/**
 * Copyright 2013-2014 Axel Huebl
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

#include <mpi.h>
#include <splash/splash.h>

#include "simulation_defines.hpp"
#include "communication/manager_common.h"
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
    class DumpHBuffer
    {
    public:
        /** Dump the PhaseSpace host Buffer
         *
         * \tparam Type the HBuffers element type
         * \tparam int the HBuffers dimension
         * \param hBuffer const reference to the hBuffer
         * \param axis_element plot to create: e.g. x, py from element_coordinate/momentum
         * \param unit sim unit of the buffer
         * \param currentStep current time step
         * \param mpiComm communicator of the participating ranks
         */
        template<typename T_Type, int T_bufDim>
        void operator()( const PMacc::container::HostBuffer<T_Type, T_bufDim>& hBuffer,
                          const std::pair<uint32_t, uint32_t> axis_element,
                          const double unit,
                          const uint32_t currentStep,
                          MPI_Comm& mpiComm ) const
        {
            using namespace splash;
            typedef T_Type Type;
            const int bufDim = T_bufDim;

            std::string filename( "phaseSpace/PhaseSpace" );

            /** get size of the fileWriter communicator ***************************/
            int size;
            MPI_CHECK(MPI_Comm_size( mpiComm, &size ));

            /** create parallel domain collector **********************************/
            ParallelDomainCollector pdc(
                mpiComm, MPI_INFO_NULL, Dimensions(size, 1, 1), 10 );

            DataCollector::FileCreationAttr fAttr;
            DataCollector::initFileCreationAttr(fAttr);
            fAttr.fileAccType = DataCollector::FAT_CREATE;

            pdc.open( filename.c_str(), fAttr );

            /** calculate global size of the phase space **************************/
            PMacc::SubGrid<simDim>& sg = Environment<simDim>::get().SubGrid();
            const size_t rOffset = sg.getSimulationBox().getGlobalOffset()[axis_element.first];
            const size_t rSize = sg.getSimulationBox().getGlobalSize()[axis_element.first];
            splash::Dimensions phaseSpace_size( rSize, hBuffer.size().y(), 1 );
            splash::Dimensions phaseSpace_global_offset( rOffset, 0, 0 );

            /** local buffer size (aka splash subdomain) **************************/
            splash::Dimensions phaseSpace_size_local( hBuffer.size().x(),
                                                          hBuffer.size().y(),
                                                          1 );

            /** Dataset Name ******************************************************/
            std::string fCoords("xyz");
            std::ostringstream dataSetName;
            /* xpx or ypz or ... */
            dataSetName << fCoords.at(axis_element.first)
                        << "p" << fCoords.at(axis_element.second);

            /** write local domain ************************************************/
            typename PICToSplash<Type>::type ctPhaseSpace;

            pdc.writeDomain( currentStep,
                             ctPhaseSpace,
                             bufDim,
                             phaseSpace_size_local,
                             dataSetName.str().c_str(),
                             phaseSpace_global_offset,
                             phaseSpace_size_local,
                             splash::Dimensions( 0, 0, 0 ),
                             phaseSpace_size,
                             DomainCollector::GridType,
                             &(*hBuffer.origin()) );

            ColTypeDouble ctDouble;
            pdc.writeAttribute( currentStep, ctDouble, dataSetName.str().c_str(),
                                "sim_unit", &unit );

            /** close file ********************************************************/
#if (SPLASH_VERSION_MAJOR>1) || ((SPLASH_VERSION_MAJOR==1) && (SPLASH_VERSION_MINOR>=2))
            pdc.finalize();
#endif
            pdc.close();
        }
    };

} /* namespace picongpu */
