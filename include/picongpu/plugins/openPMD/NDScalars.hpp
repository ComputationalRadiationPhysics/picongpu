/* Copyright 2016-2021 Alexander Grund, Franz Poeschel
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

#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/plugins/openPMD/openPMDVersion.def"

#include <pmacc/Environment.hpp>
#include <pmacc/types.hpp>

#include <stdexcept>
#include <tuple>
#include <utility>

namespace picongpu
{
    namespace openPMD
    {
        /** Functor for writing N-dimensional scalar fields with N=simDim
         * In the current implementation each process (of the ND grid of processes)
         * writes 1 scalar value Optionally the processes can also write an
         * attribute for this dataset by using a non-empty attrName
         *
         * @tparam T_Scalar    Type of the scalar value to write
         * @tparam T_Attribute Type of the attribute (can be omitted if attribute is
         * not written, defaults to uint64_t)
         */
        template<typename T_Scalar, typename T_Attribute = uint64_t>
        struct WriteNDScalars
        {
            WriteNDScalars(
                const std::string& baseName,
                const std::string& group,
                const std::string& dataset,
                const std::string& attrName = "")
                : baseName(baseName)
                , group(group)
                , dataset(dataset)
                , attrName(attrName)
            {
            }

        private:
            /** Prepare the write operation:
             *  Define openPMD dataset and write
             * attribute (if attrName is non-empty)
             *
             *  Must be called before executing the functor
             */
            std::tuple<::openPMD::MeshRecordComponent, ::openPMD::Offset, ::openPMD::Extent> prepare(
                ThreadParams& params,
                T_Attribute attribute)
            {
                auto name = baseName + "/" + group + "/" + dataset;
                const auto openPMDScalarType = ::openPMD::determineDatatype<T_Scalar>();
                using Dimensions = pmacc::math::UInt64<simDim>;

                log<picLog::INPUT_OUTPUT>("openPMD: prepare write %1%D scalars: %2%") % simDim % name;

                // Size over all processes
                Dimensions globalDomainSize = Dimensions::create(1);
                Dimensions localDomainOffset = Dimensions::create(0);

                for(uint32_t d = 0; d < simDim; ++d)
                {
                    globalDomainSize[d] = Environment<simDim>::get().GridController().getGpuNodes()[d];
                    localDomainOffset[d] = Environment<simDim>::get().GridController().getPosition()[d];
                }

                ::openPMD::Series& series = *params.openPMDSeries;
                ::openPMD::MeshRecordComponent mrc
                    = series.WRITE_ITERATIONS[params.currentStep].meshes[baseName + "_" + group][dataset];

                if(!attrName.empty())
                {
                    log<picLog::INPUT_OUTPUT>("openPMD: write attribute %1% of %2%D scalars: %3%") % attrName % simDim
                        % name;

                    mrc.setAttribute(attrName, attribute);
                }

                std::string datasetName = series.meshesPath() + baseName + "_" + group + "/" + dataset;
                params.initDataset<simDim>(
                    mrc,
                    openPMDScalarType,
                    std::move(globalDomainSize),
                    true,
                    params.compressionMethod,
                    datasetName);

                return std::make_tuple(
                    std::move(mrc),
                    static_cast<::openPMD::Offset>(asStandardVector(std::move(localDomainOffset))),
                    static_cast<::openPMD::Extent>(asStandardVector(Dimensions::create(1))));
            }

        public:
            void operator()(ThreadParams& params, T_Scalar value, T_Attribute attribute = T_Attribute())
            {
                auto tuple = prepare(params, std::move(attribute));
                auto name = baseName + "/" + group + "/" + dataset;
                log<picLog::INPUT_OUTPUT>("openPMD: write %1%D scalars: %2%") % simDim % name;

                std::get<0>(tuple).storeChunk(
                    std::make_shared<T_Scalar>(value),
                    std::move(std::get<1>(tuple)),
                    std::move(std::get<2>(tuple)));
                params.openPMDSeries->flush();
            }

        private:
            const std::string baseName, group, dataset, attrName;
            int64_t varId;
        };

        /** Functor for reading ND scalar fields with N=simDim
         * In the current implementation each process (of the ND grid of processes)
         * reads 1 scalar value Optionally the processes can also read an attribute
         * for this dataset by using a non-empty attrName
         *
         * @tparam T_Scalar    Type of the scalar value to read
         * @tparam T_Attribute Type of the attribute (can be omitted if attribute is
         * not read, defaults to uint64_t)
         */
        template<typename T_Scalar, typename T_Attribute = uint64_t>
        struct ReadNDScalars
        {
            /** Read the skalar field and optionally the attribute into the values
             * referenced by the pointers */
            void operator()(
                ThreadParams& params,
                const std::string& baseName,
                const std::string& group,
                const std::string& dataset,
                T_Scalar* value,
                const std::string& attrName = "",
                T_Attribute* attribute = nullptr)
            {
                auto name = baseName + "/" + group + "/" + dataset;
                log<picLog::INPUT_OUTPUT>("openPMD: read %1%D scalars: %2%") % simDim % name;


                auto datasetName = baseName + "/" + group + "/" + dataset;
                ::openPMD::Series& series = *params.openPMDSeries;
                ::openPMD::MeshRecordComponent mrc
                    = series.iterations[params.currentStep].meshes[baseName + "_" + group][dataset];
                auto ndim = mrc.getDimensionality();
                if(ndim != simDim)
                {
                    throw std::runtime_error(std::string("Invalid dimensionality for ") + name);
                }

                DataSpace<simDim> gridPos = Environment<simDim>::get().GridController().getPosition();
                ::openPMD::Offset start;
                ::openPMD::Extent count;
                start.reserve(ndim);
                count.reserve(ndim);
                for(int d = 0; d < ndim; ++d)
                {
                    start.push_back(gridPos.revert()[d]);
                    count.push_back(1);
                }

                __getTransactionEvent().waitForFinished();

                log<picLog::INPUT_OUTPUT>("openPMD: Schedule read scalar %1%)") % datasetName;

                std::shared_ptr<T_Scalar> readValue = mrc.loadChunk<T_Scalar>(start, count);

                series.flush();

                *value = *readValue;

                if(!attrName.empty())
                {
                    log<picLog::INPUT_OUTPUT>("openPMD: read attribute %1% for scalars: %2%") % attrName % name;
                    *attribute = mrc.getAttribute(attrName).get<T_Attribute>();
                }
            }
        };

    } // namespace openPMD
} // namespace picongpu
