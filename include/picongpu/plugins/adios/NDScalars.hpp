/* Copyright 2016-2021 Alexander Grund
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
#include "picongpu/plugins/adios/ADIOSWriter.def"
#include "picongpu/plugins/adios/restart/ReadAttribute.hpp"
#include "picongpu/traits/PICToAdios.hpp"
#include <pmacc/Environment.hpp>
#include <stdexcept>

namespace picongpu
{
    namespace adios
    {
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
            WriteNDScalars(const std::string& name, const std::string& attrName = "") : name(name), attrName(attrName)
            {
            }

            /** Prepare the write operation:
             *  Define ADIOS variable, increase params.adiosGroupSize and write attribute (if attrName is non-empty)
             *
             *  Must be called before executing the functor
             */
            void prepare(ThreadParams& params, T_Attribute attribute = T_Attribute())
            {
                typedef traits::PICToAdios<T_Scalar> AdiosSkalarType;
                typedef pmacc::math::UInt64<simDim> Dimensions;

                log<picLog::INPUT_OUTPUT>("ADIOS: prepare write %1%D scalars: %2%") % simDim % name;

                params.adiosGroupSize += sizeof(T_Scalar);
                if(!attrName.empty())
                    params.adiosGroupSize += sizeof(T_Attribute);

                // Size over all processes
                Dimensions globalDomainSize = Dimensions::create(1);
                // Offset for this process
                Dimensions localDomainOffset = Dimensions::create(0);

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    globalDomainSize[d] = Environment<simDim>::get().GridController().getGpuNodes()[d];
                    localDomainOffset[d] = Environment<simDim>::get().GridController().getPosition()[d];
                }

                std::string datasetName = params.adiosBasePath + name;

                varId = defineAdiosVar<simDim>(
                    params.adiosGroupHandle,
                    datasetName.c_str(),
                    nullptr,
                    AdiosSkalarType().type,
                    Dimensions::create(1),
                    globalDomainSize,
                    localDomainOffset,
                    true,
                    params.adiosCompression);

                if(!attrName.empty())
                {
                    typedef traits::PICToAdios<T_Attribute> AdiosAttrType;

                    log<picLog::INPUT_OUTPUT>("ADIOS: write attribute %1% of %2%D scalars: %3%") % attrName % simDim
                        % name;
                    ADIOS_CMD(adios_define_attribute_byvalue(
                        params.adiosGroupHandle,
                        attrName.c_str(),
                        datasetName.c_str(),
                        AdiosAttrType().type,
                        1,
                        (void*) &attribute));
                }
            }

            void operator()(ThreadParams& params, T_Scalar value)
            {
                log<picLog::INPUT_OUTPUT>("ADIOS: write %1%D scalars: %2%") % simDim % name;

                ADIOS_CMD(adios_write_byid(params.adiosFileHandle, varId, &value));
            }

        private:
            const std::string name, attrName;
            int64_t varId;
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
            /** Read the skalar field and optionally the attribute into the values referenced by the pointers */
            void operator()(
                ThreadParams& params,
                const std::string& name,
                T_Scalar* value,
                const std::string& attrName = "",
                T_Attribute* attribute = nullptr)
            {
                log<picLog::INPUT_OUTPUT>("ADIOS: read %1%D scalars: %2%") % simDim % name;
                std::string datasetName = params.adiosBasePath + name;

                ADIOS_VARINFO* varInfo;
                ADIOS_CMD_EXPECT_NONNULL(varInfo = adios_inq_var(params.fp, datasetName.c_str()));
                if(varInfo->ndim != simDim)
                    throw std::runtime_error(std::string("Invalid dimensionality for ") + name);
                if(varInfo->type != traits::PICToAdios<T_Scalar>().type)
                    throw std::runtime_error(std::string("Invalid type for ") + name);

                DataSpace<simDim> gridPos = Environment<simDim>::get().GridController().getPosition();
                uint64_t start[varInfo->ndim];
                uint64_t count[varInfo->ndim];
                for(int d = 0; d < varInfo->ndim; ++d)
                {
                    /* \see adios_define_var: z,y,x in C-order */
                    start[d] = gridPos.revert()[d];
                    count[d] = 1;
                }

                ADIOS_SELECTION* fSel = adios_selection_boundingbox(varInfo->ndim, start, count);

                // avoid deadlock between not finished pmacc tasks and mpi calls in adios
                __getTransactionEvent().waitForFinished();

                /* specify what we want to read, but start reading at below at `adios_perform_reads` */
                /* magic parameters (0, 1): `from_step` (not used in streams), `nsteps` to read (must be 1 for stream)
                 */
                log<picLog::INPUT_OUTPUT>("ADIOS: Schedule read skalar %1%)") % datasetName;
                ADIOS_CMD(adios_schedule_read(params.fp, fSel, datasetName.c_str(), 0, 1, (void*) value));

                /* start a blocking read of all scheduled variables */
                ADIOS_CMD(adios_perform_reads(params.fp, 1));

                adios_selection_delete(fSel);
                adios_free_varinfo(varInfo);

                if(!attrName.empty())
                {
                    log<picLog::INPUT_OUTPUT>("ADIOS: read attribute %1% for scalars: %2%") % attrName % name;
                    *attribute = readAttribute<T_Attribute>(params.fp, datasetName, attrName);
                }
            }
        };

    } // namespace adios
} // namespace picongpu
