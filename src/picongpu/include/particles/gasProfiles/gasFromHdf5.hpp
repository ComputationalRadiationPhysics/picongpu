/**
 * Copyright 2014 Felix Schmitt
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

#include "types.h"
#include "simulation_defines.hpp"
#include "memory/buffers/GridBuffer.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "simulationControl/MovingWindow.hpp"

#include <splash/splash.h>

using namespace splash;

namespace picongpu
{
    namespace gasFromHdf5
    {

        template<class Type>
        bool gasSetup(GridBuffer<Type, simDim> &fieldBuffer)
        {
            GridController<simDim> &gc = GridController<simDim>::getInstance();
            const uint32_t maxOpenFilesPerNode = 4;

            Dimensions mpiSizeHdf5(1, 1, 1);
            for (uint32_t i = 0; i < simDim; ++i)
                mpiSizeHdf5[i] = gc.getGpuNodes()[i];
                
            /* get a new ParallelDataCollector for our MPI rank and size*/
            ParallelDataCollector pdc(
                gc.getCommunicator().getMPIComm(),
                gc.getCommunicator().getMPIInfo(),
                mpiSizeHdf5,
                maxOpenFilesPerNode);

            try
            {
                /* setup ParallelDataCollector pdc to read the density information from hdf5 */
                DataSpace<simDim> mpiPos = gc.getPosition();
                DataSpace<simDim> mpiSize = gc.getGpuNodes();

                Dimensions splash_mpiPos(0, 0, 0);
                Dimensions splash_mpiSize(1, 1, 1);
                for (uint32_t i = 0; i < simDim; ++i)
                {
                    splash_mpiPos[i] = mpiPos[i];
                    splash_mpiSize[i] = mpiSize[i];
                }

                DataCollector::FileCreationAttr attr;
                DataCollector::initFileCreationAttr(attr);
                attr.fileAccType = DataCollector::FAT_READ;
                attr.mpiSize.set(splash_mpiSize);
                attr.mpiPosition.set(splash_mpiPos);

                pdc.open(gasHdf5Filename, attr);

                /* set which part of the hdf5 file our MPI rank reads */
                VirtualWindow window = MovingWindow::getInstance().getVirtualWindow(0);

                /* globalSlideOffset due to gpu slides between origin at time step 0
                 * and origin at current time step
                 * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
                 */
                DataSpace<simDim> globalSlideOffset;
                globalSlideOffset.y() = window.slides * window.localFullSize.y();

                DataSpace<simDim> globalOffset(SubGrid<simDim>::getInstance().
                    getSimulationBox().getGlobalOffset());

                Dimensions domainOffset(0, 0, 0);
                for (uint32_t d = 0; d < simDim; ++d)
                    domainOffset[d] = globalOffset[d] + globalSlideOffset[d];

                if (GridController<simDim>::getInstance().getPosition().y() == 0)
                    domainOffset[1] += window.globalSimulationOffset.y();

                DataSpace<simDim> localDomainSize = window.localSize;
                Dimensions domainSize(1, 1, 1);
                for (uint32_t d = 0; d < simDim; ++d)
                    domainSize[d] = localDomainSize[d];

                /* clear host buffer */
                fieldBuffer.getHostBuffer().setValue(float1_X(0.));
                
                /* get dimensions */
                DataSpace<simDim> fieldNoGuards = fieldBuffer.getGridLayout().getDataSpaceWithoutGuarding();
                DataSpace<simDim> guards = fieldBuffer.getGridLayout().getGuard();
                int bufferSize = fieldNoGuards.productOfComponents();

                PMACC_AUTO(dataBox, fieldBuffer.getHostBuffer().getDataBox());
                typedef DataBoxDim1Access< typename GridBuffer<Type, simDim >::DataBoxType > D1Box;
                
                /* get a 1D access object to the databox which ignores guarding */
                D1Box d1RAccess(dataBox.shift(guards), fieldNoGuards);
                
                /* allocate temporary buffer for hdf5 data */
                typedef typename Type::type ValueType;
                ValueType *tmpBfr = new ValueType[bufferSize];
                
                Dimensions sizeRead(0, 0, 0);
                pdc.read(
                        gasHdf5Iteration,
                        domainSize,
                        domainOffset,
                        gasHdf5Dataset,
                        sizeRead,
                        tmpBfr);

                pdc.close();
                
                if (sizeRead.getScalarSize() != (size_t)bufferSize)
                {
                    return false;
                }
                
                /* copy from temporary buffer to fieldTmp host buffer */
                for (int i = 0; i < bufferSize; ++i)
                {
                    d1RAccess[i].x() = tmpBfr[i];
                }
                
                __delete(tmpBfr);

                /* copy host data to the device */
                fieldBuffer.hostToDevice();
                __getTransactionEvent().waitForFinished();

            } catch (DCException e)
            {
                std::cerr << e.what() << std::endl;
                return false;
            }

            return true;
        }

        /** Load the gas density from fieldTmp
         * 
         * @param pos as DIM-D length vector offset to global left top front cell
         * @param cellIdx local cell id
         * @param fieldTmp DataBox accessing for fieldTmp
         * @return float_X between 0.0 and 1.0
         */
        template<unsigned DIM, typename FieldBox>
        DINLINE float_X calcNormedDensity(floatD_X pos, const DataSpace<DIM>& cellIdx,
                FieldBox fieldTmp)
        {
            return precisionCast<float_X>(fieldTmp(cellIdx).x());
        }
    }
}
