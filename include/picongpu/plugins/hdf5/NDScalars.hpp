/* Copyright 2016-2019 Alexander Grund
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

#include <pmacc/types.hpp>
#include "picongpu/plugins/hdf5/HDF5Writer.def"
#include "picongpu/traits/PICToSplash.hpp"
#include <pmacc/Environment.hpp>

namespace picongpu {
namespace hdf5 {

/** Functor for writing ND scalar fields with N=simDim
 * In the current implementation each process (of the ND grid of processes) writes 1 scalar value
 * Optionally the processes can also write an attribute for this dataset by using a non-empty attrName
 *
 * @tparam T_Scalar    Type of the scalar value to write
 * @tparam T_Attribute Type of the attribute (can be omitted if attribute is not written, defaults to uint64_t)
 */
template<typename T_Scalar, typename T_Attribute = uint64_t>
struct WriteNDScalars
{
    void operator()(ThreadParams& params,
            const std::string& name, T_Scalar value,
            const std::string& attrName = "", T_Attribute attribute = T_Attribute())
    {
        log<picLog::INPUT_OUTPUT>("HDF5: write %1%D scalars: %2%") % simDim % name;

        // Size over all processes
        Dimensions globalSize(1, 1, 1);
        // Offset for this process
        Dimensions localOffset(0, 0, 0);
        // Offset for all processes
        Dimensions globalOffset(0, 0, 0);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            globalSize[d] = Environment<simDim>::get().GridController().getGpuNodes()[d];
            localOffset[d] = Environment<simDim>::get().GridController().getPosition()[d];
        }

        Dimensions localSize(1, 1, 1);

        // avoid deadlock between not finished pmacc tasks and mpi calls in adios
        __getTransactionEvent().waitForFinished();

        typename traits::PICToSplash<T_Scalar>::type splashType;
        params.dataCollector->writeDomain(params.currentStep,            /* id == time step */
                                           globalSize,                   /* total size of dataset over all processes */
                                           localOffset,                  /* write offset for this process */
                                           splashType,                   /* data type */
                                           simDim,                       /* NDims spatial dimensionality of the field */
                                           splash::Selection(localSize), /* data size of this process */
                                           name.c_str(),                 /* data set name */
                                           splash::Domain(
                                                  globalOffset,          /* offset of the global domain */
                                                  globalSize             /* size of the global domain */
                                           ),
                                           DomainCollector::GridType,
                                           &value);

        if(!attrName.empty())
        {
            /*simulation attribute for data*/
            typename traits::PICToSplash<T_Attribute>::type attType;

            log<picLog::INPUT_OUTPUT>("HDF5: write attribute %1% for scalars: %2%") % attrName % name;
            params.dataCollector->writeAttribute(params.currentStep,
                                                  attType, name.c_str(),
                                                  attrName.c_str(), &attribute);
        }
    }
};

/** Functor for reading ND scalar fields with N=simDim
 * In the current implementation each process (of the ND grid of processes) reads 1 scalar value
 * Optionally the processes can also read an attribute for this dataset by using a non-empty attrName
 *
 * @tparam T_Scalar    Type of the scalar value to read
 * @tparam T_Attribute Type of the attribute (can be omitted if attribute is not read, defaults to uint64_t)
 */
template<typename T_Scalar, typename T_Attribute = uint64_t>
struct ReadNDScalars
{
    void operator()(ThreadParams& params,
                const std::string& name, T_Scalar* value,
                const std::string& attrName = "", T_Attribute* attribute = nullptr)
    {
        log<picLog::INPUT_OUTPUT>("HDF5: read %1%D scalars: %2%") % simDim % name;

        Dimensions domain_offset(0, 0, 0);
        for (uint32_t d = 0; d < simDim; ++d)
            domain_offset[d] = Environment<simDim>::get().GridController().getPosition()[d];

        // avoid deadlock between not finished pmacc tasks and mpi calls in adios
        __getTransactionEvent().waitForFinished();

        DomainCollector::DomDataClass data_class;
        DataContainer *dataContainer =
            params.dataCollector->readDomain(params.currentStep,
                                               name.c_str(),
                                               Domain(domain_offset, Dimensions(1, 1, 1)),
                                               &data_class);

        typename traits::PICToSplash<T_Scalar>::type splashType;
        *value = *static_cast<T_Scalar*>(dataContainer->getIndex(0)->getData());
        __delete(dataContainer);

        if(!attrName.empty())
        {
            log<picLog::INPUT_OUTPUT>("HDF5: read attribute %1% for scalars: %2%") % attrName % name;
            params.dataCollector->readAttributeInfo(params.currentStep, name.c_str(), attrName.c_str()).read(attribute, sizeof(T_Attribute));
            log<picLog::INPUT_OUTPUT>("HDF5: attribute %1% = %2%") % attrName % *attribute;
        }
    }
};

}  // namespace hdf5
}  // namespace picongpu
