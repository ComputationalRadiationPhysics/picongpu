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

            /** file name *****************************************************
             *    phaseSpace/PhaseSpace_xpy_timestep.h5                       */
            std::string fCoords("xyz");
            std::ostringstream filename;
            filename << "phaseSpace/PhaseSpace_"
                     << fCoords.at(axis_element.first)
                     << "p" << fCoords.at(axis_element.second);

            /** get size of the fileWriter communicator ***********************/
            int size;
            MPI_CHECK(MPI_Comm_size( mpiComm, &size ));

            /** create parallel domain collector ******************************/
            ParallelDomainCollector pdc(
                mpiComm, MPI_INFO_NULL, Dimensions(size, 1, 1), 10 );

            PMacc::GridController<simDim>& gc =
                PMacc::Environment<simDim>::get().GridController();
            DataCollector::FileCreationAttr fAttr;
            Dimensions mpiPosition( gc.getPosition()[axis_element.first], 0, 0 );
            fAttr.mpiPosition.set( mpiPosition );

            DataCollector::initFileCreationAttr(fAttr);

            pdc.open( filename.str().c_str(), fAttr );

            /** calculate global size of the phase space **********************/
            PMACC_AUTO( simBox, Environment<simDim>::get().SubGrid().getSimulationBox( ) );
            const size_t rOffset = simBox.getGlobalOffset()[axis_element.first];
            const size_t rSize = simBox.getGlobalSize()[axis_element.first];

            /* globalDomain of the phase space */
            splash::Dimensions globalPhaseSpace_size( rSize, hBuffer.size().y(), 1 );
            /* moving window meta information */
            splash::Dimensions globalPhaseSpace_offset( 0, 0, 0 );
            if( axis_element.first == 1 ) /* spatial axis == y */
            {
                VirtualWindow window = MovingWindow::getInstance( ).getVirtualWindow( currentStep );
                globalPhaseSpace_offset.set( window.slides * simBox.getLocalSize( ).y(),
                                             0, 0 );
            }

            /* localDomain: offset of it in the globalDomain and size */
            splash::Dimensions localPhaseSpace_offset( rOffset, 0, 0 );
            splash::Dimensions localPhaseSpace_size( hBuffer.size().x(),
                                                     hBuffer.size().y(),
                                                     1 );

            /** Dataset Name **************************************************/
            std::ostringstream dataSetName;
            /* xpx or ypz or ... */
            dataSetName << fCoords.at(axis_element.first)
                        << "p" << fCoords.at(axis_element.second);

            /** write local domain ********************************************/
            typename PICToSplash<Type>::type ctPhaseSpace;

            std::cout << "ps dump my start: " << rOffset << std::endl;

            pdc.writeDomain( currentStep,
                             /* global domain and my local offset within it */
                             globalPhaseSpace_size,
                             localPhaseSpace_offset,
                             /* */
                             ctPhaseSpace,
                             bufDim,
                             /* local data set dimensions */
                             Selection(localPhaseSpace_size),
                             /* data set name */
                             dataSetName.str().c_str(),
                             /* global domain */
                             Domain(
                                    globalPhaseSpace_offset,
                                    globalPhaseSpace_size
                             ),
                             /* dataClass, buffer */
                             DomainCollector::GridType,
                             &(*hBuffer.origin()) );

            ColTypeDouble ctDouble;
            pdc.writeAttribute( currentStep, ctDouble, dataSetName.str().c_str(),
                                "sim_unit", &unit );

            /** close file ****************************************************/
#if (SPLASH_VERSION_MAJOR>1) || ((SPLASH_VERSION_MAJOR==1) && (SPLASH_VERSION_MINOR>=2))
            pdc.finalize();
#endif
            pdc.close();
        }
    };

} /* namespace picongpu */
