/* Copyright 2013-2019 Axel Huebl, Rene Widera
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/traits/SplashToPIC.hpp"
#include "picongpu/traits/PICToSplash.hpp"

#include "picongpu/plugins/PhaseSpace/AxisDescription.hpp"
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/cuSTL/container/HostBuffer.hpp>
#include <pmacc/math/vector/Int.hpp>
#include <pmacc/verify.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <utility>
#include <mpi.h>
#include <splash/splash.h>

namespace picongpu
{
    class DumpHBuffer
    {
    private:
       typedef typename MappingDesc::SuperCellSize SuperCellSize;

    public:
        /** Dump the PhaseSpace host Buffer
         *
         * \tparam Type the HBuffers element type
         * \tparam int the HBuffers dimension
         * \param hBuffer const reference to the hBuffer, including guard cells in spatial dimension
         * \param axis_element plot to create: e.g. py, x from momentum/spatial-coordinate
         * \param unit sim unit of the buffer
         * \param strSpecies unique short hand name of the species
         * \param currentStep current time step
         * \param mpiComm communicator of the participating ranks
         */
        template<typename T_Type, int T_bufDim>
        void operator()( const pmacc::container::HostBuffer<T_Type, T_bufDim>& hBuffer,
                         const AxisDescription axis_element,
                         const std::pair<float_X, float_X> axis_p_range,
                         const float_64 pRange_unit,
                         const float_64 unit,
                         const std::string strSpecies,
                         const uint32_t currentStep,
                         MPI_Comm mpiComm ) const
        {
            using namespace splash;
            typedef T_Type Type;
            const int bufDim = T_bufDim;

            /** file name *****************************************************
             *    phaseSpace/PhaseSpace_xpy_timestep.h5                       */
            std::string fCoords("xyz");
            std::ostringstream filename;
            filename << "phaseSpace/PhaseSpace_"
                     << strSpecies << "_"
                     << fCoords.at(axis_element.space)
                     << "p" << fCoords.at(axis_element.momentum);

            /** get size of the fileWriter communicator ***********************/
            int size;
            MPI_CHECK(MPI_Comm_size( mpiComm, &size ));

            /** create parallel domain collector ******************************/
            ParallelDomainCollector pdc(
                mpiComm, MPI_INFO_NULL, Dimensions(size, 1, 1), 10 );

            pmacc::GridController<simDim>& gc =
                pmacc::Environment<simDim>::get().GridController();
            DataCollector::FileCreationAttr fAttr;
            Dimensions mpiPosition( gc.getPosition()[axis_element.space], 0, 0 );
            fAttr.mpiPosition.set( mpiPosition );

            DataCollector::initFileCreationAttr(fAttr);

            pdc.open( filename.str().c_str(), fAttr );

            /** calculate GUARD offset in the source hBuffer *****************/
            const uint32_t rGuardCells =
                SuperCellSize().toRT()[axis_element.space] * GuardSize::toRT()[axis_element.space];

            /** calculate local and global size of the phase space ***********/
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const int rLocalOffset = subGrid.getLocalDomain().offset[axis_element.space];
            const int rLocalSize = int(hBuffer.size().y() - 2*rGuardCells);
            const int rGlobalSize = subGrid.getGlobalDomain().size[axis_element.space];
            PMACC_VERIFY( rLocalSize == subGrid.getLocalDomain().size[axis_element.space] );

            /* globalDomain of the phase space */
            splash::Dimensions globalPhaseSpace_size( hBuffer.size().x(),
                                                      rGlobalSize,
                                                      1 );

            /* global moving window meta information */
            splash::Dimensions globalPhaseSpace_offset( 0, 0, 0 );
            int globalMovingWindowOffset = 0;
            int globalMovingWindowSize   = rGlobalSize;
            if( axis_element.space == AxisDescription::y ) /* spatial axis == y */
            {
                globalPhaseSpace_offset.set( 0, numSlides * rLocalSize, 0 );
                Window window = MovingWindow::getInstance( ).getWindow( currentStep );
                globalMovingWindowOffset = window.globalDimensions.offset[axis_element.space];
                globalMovingWindowSize = window.globalDimensions.size[axis_element.space];
            }

            /* localDomain: offset of it in the globalDomain and size */
            splash::Dimensions localPhaseSpace_offset( 0, rLocalOffset, 0 );
            splash::Dimensions localPhaseSpace_size( hBuffer.size().x(),
                                                     rLocalSize,
                                                     1 );

            /** Dataset Name **************************************************/
            std::ostringstream dataSetName;
            /* xpx or ypz or ... */
            dataSetName << fCoords.at(axis_element.space)
                        << "p" << fCoords.at(axis_element.momentum);

            /** debug log *****************************************************/
            int rank;
            MPI_CHECK(MPI_Comm_rank( mpiComm, &rank ));
            log<picLog::INPUT_OUTPUT > ("Dump buffer %1% to %2% at offset %3% with size %4% for total size %5% for rank %6% / %7%")
                % ( *(hBuffer.origin()(0,rGuardCells)) ) % dataSetName.str() % localPhaseSpace_offset.toString()
                % localPhaseSpace_size.toString() % globalPhaseSpace_size.toString()
                % rank % size;

            /** write local domain ********************************************/
            typename PICToSplash<Type>::type ctPhaseSpace;

            // avoid deadlock between not finished pmacc tasks and mpi calls in HDF5
            __getTransactionEvent().waitForFinished();

            pdc.writeDomain( currentStep,
                             /* global domain and my local offset within it */
                             globalPhaseSpace_size,
                             localPhaseSpace_offset,
                             /* */
                             ctPhaseSpace,
                             bufDim,
                             /* local data set dimensions */
                             splash::Selection(localPhaseSpace_size),
                             /* data set name */
                             dataSetName.str().c_str(),
                             /* global domain */
                             splash::Domain(
                                    globalPhaseSpace_offset,
                                    globalPhaseSpace_size
                             ),
                             /* dataClass, buffer */
                             DomainCollector::GridType,
                             &(*hBuffer.origin()(0,rGuardCells)) );

            /** meta attributes for the data set: unit, range, moving window **/
            typedef PICToSplash<float_X>::type  SplashFloatXType;
            typedef PICToSplash<float_64>::type SplashFloat64Type;
            ColTypeInt ctInt;
            SplashFloat64Type ctFloat64;
            SplashFloatXType  ctFloatX;

            pdc.writeAttribute( currentStep, ctFloat64, dataSetName.str().c_str(),
                                "sim_unit", &unit );
            pdc.writeAttribute( currentStep, ctFloat64, dataSetName.str().c_str(),
                                "p_unit", &pRange_unit );
            pdc.writeAttribute( currentStep, ctFloatX, dataSetName.str().c_str(),
                                "p_min", &(axis_p_range.first) );
            pdc.writeAttribute( currentStep, ctFloatX, dataSetName.str().c_str(),
                                "p_max", &(axis_p_range.second) );
            pdc.writeAttribute( currentStep, ctInt, dataSetName.str().c_str(),
                                "movingWindowOffset", &globalMovingWindowOffset );
            pdc.writeAttribute( currentStep, ctInt, dataSetName.str().c_str(),
                                "movingWindowSize", &globalMovingWindowSize );

            pdc.writeAttribute( currentStep, ctFloatX, dataSetName.str().c_str(),
                                "dr", &(cellSize[axis_element.space]) );
            pdc.writeAttribute( currentStep, ctFloatX, dataSetName.str().c_str(),
                                "dV", &CELL_VOLUME );
            pdc.writeAttribute( currentStep, ctFloat64, dataSetName.str().c_str(),
                                "dr_unit", &UNIT_LENGTH );
            pdc.writeAttribute( currentStep, ctFloatX, dataSetName.str().c_str(),
                                "dt", &DELTA_T );
            pdc.writeAttribute( currentStep, ctFloat64, dataSetName.str().c_str(),
                                "dt_unit", &UNIT_TIME );

            /** close file ****************************************************/
            pdc.finalize();
            pdc.close();
        }
    };

} /* namespace picongpu */
