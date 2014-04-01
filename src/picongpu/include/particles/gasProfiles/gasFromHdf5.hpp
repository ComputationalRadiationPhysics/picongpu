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
        bool gasSetup(GridBuffer<Type, simDim> &fieldBuffer, VirtualWindow &window)
        {
            GridController<simDim> &gc = Environment<simDim>::get().GridController();
            const uint32_t maxOpenFilesPerNode = 1;

            /* get a new ParallelDomainCollector for our MPI rank only*/
            ParallelDomainCollector pdc(
                    MPI_COMM_SELF,
                    gc.getCommunicator().getMPIInfo(),
                    Dimensions(1, 1, 1),
                    maxOpenFilesPerNode);

            try
            {
                /* setup ParallelDomainCollector pdc to read the density information from hdf5 */
                DataCollector::FileCreationAttr attr;
                DataCollector::initFileCreationAttr(attr);
                attr.fileAccType = DataCollector::FAT_READ;

                pdc.open(gasHdf5Filename, attr);

                /* set which part of the hdf5 file our MPI rank reads */
                DataSpace<simDim> globalSlideOffset;
                globalSlideOffset.y() = window.slides * window.localFullSize.y();

                DataSpace<simDim> globalOffset(Environment<simDim>::get().SubGrid().
                        getSimulationBox().getGlobalOffset());

                Dimensions domainOffset(0, 0, 0);
                for (uint32_t d = 0; d < simDim; ++d)
                    domainOffset[d] = globalOffset[d] + globalSlideOffset[d];

                if (gc.getPosition().y() == 0)
                    domainOffset[1] += window.globalSimulationOffset.y();

                DataSpace<simDim> localDomainSize = window.localFullSize;
                Dimensions domainSize(1, 1, 1);
                for (uint32_t d = 0; d < simDim; ++d)
                    domainSize[d] = localDomainSize[d];

                /* clear host buffer with default value */
                fieldBuffer.getHostBuffer().setValue(float1_X(gasDefaultValue));

                /* get dimensions and offsets (collective call) */
                Domain fileDomain = pdc.getGlobalDomain(gasHdf5Iteration, gasHdf5Dataset);
                Dimensions fileDomainEnd = fileDomain.getOffset() + fileDomain.getSize();
                DataSpace<simDim> accessSpace;
                DataSpace<simDim> accessOffset;

                Dimensions fileAccessSpace(1, 1, 1);
                Dimensions fileAccessOffset(0, 0, 0);

                /* For each dimension, compute how file domain and local simulation domain overlap
                 * and which sizes and offsets are required for loading data from the file. 
                 **/
                for (uint32_t d = 0; d < simDim; ++d)
                {
                    /* file domain in/in-after sim domain */
                    if (fileDomain.getOffset()[d] >= domainOffset[d] &&
                            fileDomain.getOffset()[d] <= domainOffset[d] + domainSize[d])
                    {
                        accessSpace[d] = std::min(domainOffset[d] + domainSize[d] - fileDomain.getOffset()[d],
                                fileDomain.getSize()[d]);
                        fileAccessSpace[d] = accessSpace[d];

                        accessOffset[d] = fileDomain.getOffset()[d] - domainOffset[d];
                        fileAccessOffset[d] = 0;
                        continue;
                    }

                    /* file domain before-in sim domain */
                    if (fileDomainEnd[d] >= domainOffset[d] &&
                            fileDomainEnd[d] <= domainOffset[d] + domainSize[d])
                    {
                        accessSpace[d] = fileDomainEnd[d] - domainOffset[d];
                        fileAccessSpace[d] = accessSpace[d];

                        accessOffset[d] = 0;
                        fileAccessOffset[d] = domainOffset[d] - fileDomain.getOffset()[d];
                        continue;
                    }

                    /* sim domain in file domain */
                    if (domainOffset[d] >= fileDomain.getOffset()[d] &&
                            domainOffset[d] + domainSize[d] <= fileDomainEnd[d])
                    {
                        accessSpace[d] = domainSize[d];
                        fileAccessSpace[d] = accessSpace[d];

                        accessOffset[d] = 0;
                        fileAccessOffset[d] = domainOffset[d] - fileDomain.getOffset()[d];
                        continue;
                    }

                    /* file domain and sim domain do not intersect, do not load anything */
                    accessSpace[d] = 0;
                    break;
                }

                /* allocate temporary buffer for hdf5 data */
                typedef typename Type::type ValueType;
                ValueType *tmpBfr = NULL;

                size_t accessSize = accessSpace.productOfComponents();
                if (accessSize > 0)
                {
                    tmpBfr = new ValueType[accessSize];

                    Dimensions sizeRead(0, 0, 0);
                    pdc.read(
                            gasHdf5Iteration,
                            fileAccessSpace,
                            fileAccessOffset,
                            gasHdf5Dataset,
                            sizeRead,
                            tmpBfr);

                    if (sizeRead.getScalarSize() != accessSize)
                    {
                        __delete(tmpBfr);
                        return false;
                    }

                    /* get the databox of the host buffer */
                    PMACC_AUTO(dataBox, fieldBuffer.getHostBuffer().getDataBox());
                    /* get a 1D access object to the databox */
                    typedef DataBoxDim1Access< typename GridBuffer<Type, simDim >::DataBoxType > D1Box;
                    DataSpace<simDim> guards = fieldBuffer.getGridLayout().getGuard();
                    D1Box d1RAccess(dataBox.shift(guards + accessOffset), accessSpace);

                    /* copy from temporary buffer to fieldTmp host buffer */
                    for (int i = 0; i < accessSpace.productOfComponents(); ++i)
                    {
                        d1RAccess[i].x() = tmpBfr[i];
                    }

                    __delete(tmpBfr);
                }
                
                pdc.close();

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
