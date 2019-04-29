/* Copyright 2014-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/hdf5/HDF5Writer.def"
#include "picongpu/traits/PICToSplash.hpp"
#include <pmacc/traits/GetComponentsType.hpp>
#include <pmacc/traits/GetNComponents.hpp>
#include <pmacc/assert.hpp>

#include <string>

namespace picongpu
{

namespace hdf5
{

using namespace pmacc;
using namespace splash;

struct Field
{

    /* \param inCellPosition std::vector<std::vector<float_X> > with the outer
     *                       vector for each component and the inner vector for
     *                       the simDim position offset within the cell [0.0; 1.0)
     */
    template<typename T_ValueType, typename T_DataBoxType>
    static void writeField(ThreadParams *params,
                           const std::string name,
                           std::vector<float_64> unit,
                           std::vector<float_64> unitDimension,
                           std::vector<std::vector<float_X> > inCellPosition,
                           float_X timeOffset,
                           T_DataBoxType dataBox,
                           const T_ValueType&
                           )
    {
        typedef T_DataBoxType NativeDataBoxType;
        typedef T_ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;
        typedef typename PICToSplash<float_X>::type SplashFloatXType;

        const uint32_t nComponents = GetNComponents<ValueType>::value;

        SplashType splashType;
        ColTypeDouble ctDouble;
        SplashFloatXType splashFloatXType;

        log<picLog::INPUT_OUTPUT > ("HDF5 write field: %1% %2%") %
            name % nComponents;

        /* parameter checking */
        PMACC_ASSERT( unit.size() == nComponents );
        PMACC_ASSERT( inCellPosition.size() == nComponents );
        for( uint32_t n = 0; n < nComponents; ++n )
            PMACC_ASSERT( inCellPosition.at(n).size() == simDim );
        PMACC_ASSERT(unitDimension.size() == 7); // seven openPMD base units

        /* component names */
        const std::string recordName = std::string("fields/") + name;

        std::vector<std::string> name_lookup;
        {
            const std::string name_lookup_tpl[] = {"x", "y", "z", "w"};
            for (uint32_t n = 0; n < nComponents; n++)
                name_lookup.push_back(name_lookup_tpl[n]);
        }

        /*data to describe source buffer*/
        GridLayout<simDim> field_layout = params->gridLayout;
        DataSpace<simDim> field_no_guard = params->window.localDimensions.size;
        DataSpace<simDim> field_guard = field_layout.getGuard() + params->localWindowToDomainOffset;
        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset;
        const pmacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(params->currentStep);
        globalSlideOffset.y() += numSlides * localDomain.size.y();

        Dimensions splashGlobalDomainOffset(0, 0, 0);
        Dimensions splashGlobalOffsetFile(0, 0, 0);
        Dimensions splashGlobalDomainSize(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            splashGlobalOffsetFile[d] = localDomain.offset[d];
            splashGlobalDomainOffset[d] = params->window.globalDimensions.offset[d] + globalSlideOffset[d];
            splashGlobalDomainSize[d] = params->window.globalDimensions.size[d];
        }

        splashGlobalOffsetFile[1] = std::max(0, localDomain.offset[1] -
                                             params->window.globalDimensions.offset[1]);

        size_t tmpArraySize = field_no_guard.productOfComponents();
        ComponentType* tmpArray = new ComponentType[tmpArraySize];

        typedef DataBoxDim1Access<NativeDataBoxType > D1Box;
        D1Box d1Access(dataBox.shift(field_guard), field_no_guard);

        for (uint32_t n = 0; n < nComponents; n++)
        {
            /* copy data to temp array
             * tmpArray has the size of the data without any offsets
             */
            for (size_t i = 0; i < tmpArraySize; ++i)
            {
                tmpArray[i] = d1Access[i][n];
            }

            std::stringstream datasetName;
            datasetName << recordName;
            if (nComponents > 1)
                datasetName << "/" << name_lookup.at(n);

            Dimensions sizeSrcData(1, 1, 1);

            for (uint32_t d = 0; d < simDim; ++d)
            {
                sizeSrcData[d] = field_no_guard[d];
            }

            // avoid deadlock between not finished pmacc tasks and mpi calls in splash/HDF5
            __getTransactionEvent().waitForFinished();
            params->dataCollector->writeDomain(params->currentStep,             /* id == time step */
                                               splashGlobalDomainSize,          /* total size of dataset over all processes */
                                               splashGlobalOffsetFile,          /* write offset for this process */
                                               splashType,                      /* data type */
                                               simDim,                          /* NDims spatial dimensionality of the field */
                                               splash::Selection(sizeSrcData),  /* data size of this process */
                                               datasetName.str().c_str(),       /* data set name */
                                               splash::Domain(
                                                      splashGlobalDomainOffset, /* offset of the global domain */
                                                      splashGlobalDomainSize    /* size of the global domain */
                                               ),
                                               DomainCollector::GridType,
                                               tmpArray);

            /* attributes */
            params->dataCollector->writeAttribute(params->currentStep,
                                                  splashFloatXType, datasetName.str().c_str(),
                                                  "position",
                                                  1u, Dimensions(simDim,0,0),
                                                  &(*inCellPosition.at(n).begin()));

            params->dataCollector->writeAttribute(params->currentStep,
                                                  ctDouble, datasetName.str().c_str(),
                                                  "unitSI", &(unit.at(n)));
        }
        __deleteArray(tmpArray);


        params->dataCollector->writeAttribute(params->currentStep,
                                              ctDouble, recordName.c_str(),
                                              "unitDimension",
                                              1u, Dimensions(7,0,0),
                                              &(*unitDimension.begin()));

        params->dataCollector->writeAttribute(params->currentStep,
                                              splashFloatXType, recordName.c_str(),
                                              "timeOffset", &timeOffset);

        const std::string geometry("cartesian");
        ColTypeString ctGeometry(geometry.length());
        params->dataCollector->writeAttribute(params->currentStep,
                                              ctGeometry, recordName.c_str(),
                                              "geometry", geometry.c_str());

        const std::string dataOrder("C");
        ColTypeString ctDataOrder(dataOrder.length());
        params->dataCollector->writeAttribute(params->currentStep,
                                              ctDataOrder, recordName.c_str(),
                                              "dataOrder", dataOrder.c_str());

        char axisLabels[simDim][2];
        ColTypeString ctAxisLabels(1);
        for( uint32_t d = 0; d < simDim; ++d )
        {
            axisLabels[simDim-1-d][0] = char('x' + d); // 3D: F[z][y][x], 2D: F[y][x]
            axisLabels[simDim-1-d][1] = '\0';          // terminator is important!
        }
        params->dataCollector->writeAttribute(params->currentStep,
                                              ctAxisLabels, recordName.c_str(),
                                              "axisLabels",
                                              1u, Dimensions(simDim,0,0),
                                              axisLabels);

        // cellSize is {x, y, z} but fields are F[z][y][x]
        std::vector<float_X> gridSpacing(simDim, 0.0);
        for( uint32_t d = 0; d < simDim; ++d )
            gridSpacing.at(simDim-1-d) = cellSize[d];
        params->dataCollector->writeAttribute(params->currentStep,
                                              splashFloatXType, recordName.c_str(),
                                              "gridSpacing",
                                              1u, Dimensions(simDim,0,0),
                                              &(*gridSpacing.begin()));

        // splashGlobalDomainOffset is {x, y, z} but fields are F[z][y][x]
        std::vector<float_64> gridGlobalOffset(simDim, 0.0);
        for( uint32_t d = 0; d < simDim; ++d )
            gridGlobalOffset.at(simDim-1-d) =
                float_64(cellSize[d]) *
                float_64(splashGlobalDomainOffset[d]);
        params->dataCollector->writeAttribute(params->currentStep,
                                              ctDouble, recordName.c_str(),
                                              "gridGlobalOffset",
                                              1u, Dimensions(simDim,0,0),
                                              &(*gridGlobalOffset.begin()));

        params->dataCollector->writeAttribute(params->currentStep,
                                              ctDouble, recordName.c_str(),
                                              "gridUnitSI", &UNIT_LENGTH);

        const std::string fieldSmoothing("none");
        ColTypeString ctFieldSmoothing(fieldSmoothing.length());
        params->dataCollector->writeAttribute(params->currentStep,
                                              ctFieldSmoothing, recordName.c_str(),
                                              "fieldSmoothing", fieldSmoothing.c_str());
    }

};

} //namspace hdf5

} //namespace picongpu
