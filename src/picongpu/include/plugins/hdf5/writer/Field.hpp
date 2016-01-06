/**
 * Copyright 2014-2016 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include "pmacc_types.hpp"
#include "simulation_types.hpp"
#include "plugins/hdf5/HDF5Writer.def"
#include "traits/PICToSplash.hpp"
#include "traits/GetComponentsType.hpp"
#include "traits/GetNComponents.hpp"

namespace picongpu
{

namespace hdf5
{

using namespace PMacc;
using namespace splash;

struct Field
{

    template<typename T_ValueType, typename T_DataBoxType>
    static void writeField(ThreadParams *params,
                           const std::string name,
                           std::vector<float_64> unit,
                           /* unitDimension, position, timeOffset */
                           T_DataBoxType dataBox,
                           const T_ValueType&
                           )
    {
        typedef T_DataBoxType NativeDataBoxType;
        typedef T_ValueType ValueType;
        typedef typename GetComponentsType<ValueType>::type ComponentType;
        typedef typename PICToSplash<ComponentType>::type SplashType;

        const uint32_t nComponents = GetNComponents<ValueType>::value;

        log<picLog::INPUT_OUTPUT > ("HDF5 write field: %1% %2%") %
            name % nComponents;

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
        const PMacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
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

        SplashType splashType;

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
            datasetName << "fields/" << name;
            if (nComponents > 1)
                datasetName << "/" << name_lookup.at(n);

            Dimensions sizeSrcData(1, 1, 1);

            for (uint32_t d = 0; d < simDim; ++d)
            {
                sizeSrcData[d] = field_no_guard[d];
            }

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

            /*simulation attributes for data*/
            ColTypeDouble ctDouble;

            params->dataCollector->writeAttribute(params->currentStep,
                                                  ctDouble, datasetName.str().c_str(),
                                                  "unitSI", &(unit.at(n)));
            /* position */
        }
        __deleteArray(tmpArray);

        std::string recordName = std::string("fields/") + name;

        /* unitDimension, timeOffset */

        std::string geometry("cartesian");
        ColTypeString ctGeometry(geometry.length());
        params->dataCollector->writeAttribute(params->currentStep,
                                              ctGeometry, recordName.c_str(),
                                              "geometry", geometry.c_str());

        std::string dataOrder("C");
        ColTypeString ctDataOrder(dataOrder.length());
        params->dataCollector->writeAttribute(params->currentStep,
                                              ctDataOrder, recordName.c_str(),
                                              "dataOrder", dataOrder.c_str());

        /* axisLabels, gridSpacing, gridGlobalOffset */

        ColTypeDouble ctDouble;
        params->dataCollector->writeAttribute(params->currentStep,
                                              ctDouble, recordName.c_str(),
                                              "gridUnitSI", &UNIT_LENGTH);
    }

};

} //namspace hdf5

} //namespace picongpu
