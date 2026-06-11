/*
 * SPDX-FileCopyrightText: Alexander Grund, Franz Poeschel
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/common/openPMDDefinitions.def"
#    include "picongpu/plugins/openPMD/openPMDWriter.def"

#    include <pmacc/Environment.hpp>
#    include <pmacc/types.hpp>

#    include <stdexcept>
#    include <tuple>
#    include <utility>

#    include <openPMD/openPMD.hpp>

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
                std::string const& baseName,
                std::string const& group,
                std::string const& dataset,
                std::string const& attrName = "")
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
                uint32_t const currentStep,
                T_Attribute attribute)
            {
                auto name = baseName + "/" + group + "/" + dataset;
                auto const openPMDScalarType = ::openPMD::determineDatatype<T_Scalar>();
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
                    = series.writeIterations()[currentStep].meshes[baseName + "_" + group][dataset];

                if(!attrName.empty())
                {
                    log<picLog::INPUT_OUTPUT>("openPMD: write attribute %1% of %2%D scalars: %3%") % attrName % simDim
                        % name;

                    mrc.setAttribute(attrName, attribute);
                }

                std::string datasetName = series.meshesPath() + baseName + "_" + group + "/" + dataset;
                params.initDataset<simDim>(mrc, openPMDScalarType, std::move(globalDomainSize), datasetName);

                return std::make_tuple(
                    std::move(mrc),
                    static_cast<::openPMD::Offset>(asStandardVector(std::move(localDomainOffset))),
                    static_cast<::openPMD::Extent>(asStandardVector(Dimensions::create(1))));
            }

        public:
            void operator()(
                ThreadParams& params,
                uint32_t const currentStep,
                T_Scalar value,
                T_Attribute attribute = T_Attribute())
            {
                auto tuple = prepare(params, currentStep, std::move(attribute));
                auto name = baseName + "/" + group + "/" + dataset;
                log<picLog::INPUT_OUTPUT>("openPMD: write %1%D scalars: %2%") % simDim % name;

                std::get<0>(tuple).storeChunk(
                    std::make_shared<T_Scalar>(value),
                    std::move(std::get<1>(tuple)),
                    std::move(std::get<2>(tuple)));
                params.openPMDSeries->flush(PreferredFlushTarget::Buffer);
            }

        private:
            std::string const baseName, group, dataset, attrName;
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
                uint32_t const currentStep,
                std::string const& baseName,
                std::string const& group,
                std::string const& dataset,
                T_Scalar* value,
                std::string const& attrName = "",
                T_Attribute* attribute = nullptr)
            {
                auto name = baseName + "/" + group + "/" + dataset;
                log<picLog::INPUT_OUTPUT>("openPMD: read %1%D scalars: %2%") % simDim % name;


                auto datasetName = baseName + "/" + group + "/" + dataset;
                ::openPMD::Series& series = *params.openPMDSeries;
                ::openPMD::MeshRecordComponent mrc
                    = series.iterations[currentStep].open().meshes[baseName + "_" + group][dataset];
                auto ndim = mrc.getDimensionality();
                if(ndim != simDim)
                {
                    throw std::runtime_error(std::string("Invalid dimensionality for ") + name);
                }

                DataSpace<simDim> gridPos = Environment<simDim>::get().GridController().getPosition();
                ::openPMD::Offset start;
                ::openPMD::Extent count;
                ::openPMD::Extent extent = mrc.getExtent();
                start.reserve(ndim);
                count.reserve(ndim);
                for(int d = 0; d < ndim; ++d)
                {
                    /*
                     * When restarting with more parallel processes than the checkpoint had originally been written
                     * with, we must take care not to index past the dataset boundaries. Just loop around to the start
                     * in that case. Not the finest way, but it does the job for now..
                     */
                    start.push_back(gridPos.revert()[d] % extent[d]);
                    count.push_back(1);
                }

                eventSystem::getTransactionEvent().waitForFinished();

                log<picLog::INPUT_OUTPUT>("openPMD: Schedule read scalar %1%)") % datasetName;

                std::shared_ptr<T_Scalar> readValue = mrc.loadChunk<T_Scalar>(start, count);

                mrc.seriesFlush();

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

#endif
