/**
 * Copyright 2016 Alexander Grund
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
#include "plugins/hdf5/HDF5Writer.def"
#include "traits/PICToSplash.hpp"
#include "Environment.hpp"

namespace picongpu {
namespace hdf5 {

/* Functors for reading and writing ND scalar fields
 * In the current implementation each process (of the ND grid) reads/writes 1 scalar value
 * Optionally the processes can also write an attribute for this dataset
 */

template<typename T_Skalar, typename T_Attribute = uint64_t>
struct WriteNDScalars
{
    void operator()(ThreadParams& params,
            const std::string& name, T_Skalar value,
            const std::string& attrName = "", T_Attribute attribute = T_Attribute())
    {
        log<picLog::INPUT_OUTPUT> ("HDF5 write %1%D scalars: %2%") % simDim % name;

        // Size over all processes
        Dimensions splashGlobalDomainSize(1, 1, 1);
        // Offset for this process
        Dimensions splashGlobalOffsetFile(0, 0, 0);
        // Offset for all processes
        Dimensions splashGlobalDomainOffset(0, 0, 0);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            splashGlobalDomainSize[d] = Environment<simDim>::get().GridController().getGpuNodes()[d];
            splashGlobalOffsetFile[d] = Environment<simDim>::get().GridController().getPosition()[d];
        }

        Dimensions localSize(1, 1, 1);

        typename traits::PICToSplash<T_Skalar>::type splashType;
        params.dataCollector->writeDomain(params.currentStep,               /* id == time step */
                                           splashGlobalDomainSize,          /* total size of dataset over all processes */
                                           splashGlobalOffsetFile,          /* write offset for this process */
                                           splashType,                      /* data type */
                                           simDim,                          /* NDims spatial dimensionality of the field */
                                           splash::Selection(localSize),    /* data size of this process */
                                           name.c_str(),                    /* data set name */
                                           splash::Domain(
                                                  splashGlobalDomainOffset, /* offset of the global domain */
                                                  splashGlobalDomainSize    /* size of the global domain */
                                           ),
                                           DomainCollector::GridType,
                                           &value);

        if(!attrName.empty())
        {
            log<picLog::INPUT_OUTPUT> ("HDF5 write attribute %1% for scalars: %2%") % attrName % name;
            /*simulation attributes for data*/
            typename traits::PICToSplash<T_Attribute>::type attType;

            params.dataCollector->writeAttribute(params.currentStep,
                                                  attType, name.c_str(),
                                                  attrName.c_str(), &attribute);
        }
    }
};

template<typename T_Skalar, typename T_Attribute = uint64_t>
struct ReadNDScalars
{
    void operator()(ThreadParams& params,
                const std::string& name, T_Skalar* value,
                const std::string& attrName = "", T_Attribute* attribute = NULL)
    {
        log<picLog::INPUT_OUTPUT> ("HDF5 read %1%D scalars: %2%") % simDim % name;

        Dimensions domain_offset(0, 0, 0);
        for (uint32_t d = 0; d < simDim; ++d)
            domain_offset[d] = Environment<simDim>::get().GridController().getPosition()[d];

        DomainCollector::DomDataClass data_class;
        DataContainer *dataContainer =
            params.dataCollector->readDomain(params.currentStep,
                                               name.c_str(),
                                               Domain(domain_offset, Dimensions(1, 1, 1)),
                                               &data_class);

        typename traits::PICToSplash<T_Skalar>::type splashType;
        *value = *static_cast<T_Skalar*>(dataContainer->getIndex(0)->getData());
        __delete(dataContainer);

        if(!attrName.empty())
        {
            log<picLog::INPUT_OUTPUT> ("HDF5 read attribute %1% for scalars: %2%") % attrName % name;
            params.dataCollector->readAttribute(params.currentStep, name.c_str(), attrName.c_str(), attribute);
        }
    }
};

}  // namespace hdf5
}  // namespace picongpu
