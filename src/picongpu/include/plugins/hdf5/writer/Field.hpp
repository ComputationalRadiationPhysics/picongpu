/**
 * Copyright 2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include "types.h"
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
                           const DomainInformation domInfo,
                           const std::string name,
                           std::vector<double> unit,
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
            for (uint32_t d = 0; d < nComponents; d++)
                name_lookup.push_back(name_lookup_tpl[d]);
        }

        /*data to describe source buffer*/
        GridLayout<simDim> field_layout = params->gridLayout;
        DataSpace<simDim> field_no_guard = domInfo.domainSize;
        DataSpace<simDim> field_guard = field_layout.getGuard() + domInfo.localDomainOffset;
        /* globalSlideOffset due to gpu slides between origin at time step 0
         * and origin at current time step
         * ATTENTION: splash offset are globalSlideOffset + picongpu offsets
         */
        DataSpace<simDim> globalSlideOffset;
        globalSlideOffset.y() += params->window.slides * params->window.localDomainSize.y();

        Dimensions splashGlobalDomainOffset(0, 0, 0);
        Dimensions splashGlobalOffsetFile(0, 0, 0);
        Dimensions splashGlobalDomainSize(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            splashGlobalOffsetFile[d] = domInfo.domainOffset[d];
            splashGlobalDomainOffset[d] = domInfo.globalDomainOffset[d] + globalSlideOffset[d];
            splashGlobalDomainSize[d] = domInfo.globalDomainSize[d];
        }

        splashGlobalOffsetFile[1] = std::max(0, domInfo.domainOffset[1] -
                                             domInfo.globalDomainOffset[1]);

        SplashType splashType;

        size_t tmpArraySize = field_no_guard.productOfComponents();
        ComponentType* tmpArray = new ComponentType[tmpArraySize];

        typedef DataBoxDim1Access<NativeDataBoxType > D1Box;
        D1Box d1Access(dataBox.shift(field_guard), field_no_guard);

        for (uint32_t d = 0; d < nComponents; d++)
        {
            /* copy data to temp array
             * tmpArray has the size of the data without any offsets
             */
            for (size_t i = 0; i < tmpArraySize; ++i)
            {
                tmpArray[i] = d1Access[i][d];
            }

            std::stringstream datasetName;
            datasetName << "fields/" << name;
            if (nComponents > 1)
                datasetName << "/" << name_lookup.at(d);

            Dimensions sizeSrcData(1, 1, 1);

            for (uint32_t i = 0; i < simDim; ++i)
            {
                sizeSrcData[i] = field_no_guard[i];
            }

            params->dataCollector->writeDomain(params->currentStep, /* id == time step */
                                               splashGlobalDomainSize,
                                               splashGlobalOffsetFile,
                                               splashType, /* data type */
                                               simDim, /* NDims of the field data (scalar, vector, ...) */
                                               sizeSrcData,
                                               datasetName.str().c_str(), /* data set name */
                                               splashGlobalDomainOffset, /* \todo offset of the global domain */
                                               splashGlobalDomainSize, /* size of the global domain */
                                               DomainCollector::GridType,
                                               tmpArray);

            /*simulation attributes for data*/
            ColTypeDouble ctDouble;

            params->dataCollector->writeAttribute(params->currentStep,
                                                  ctDouble, datasetName.str().c_str(),
                                                  "sim_unit", &(unit.at(d)));
        }
        __deleteArray(tmpArray);
    }

};

} //namspace hdf5

} //namespace picongpu
