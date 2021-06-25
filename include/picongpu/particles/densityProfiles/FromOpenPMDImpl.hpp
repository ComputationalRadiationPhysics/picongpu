/* Copyright 2013-2021 Felix Schmitt, Rene Widera, Sergei Bastrakov
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

#if(ENABLE_OPENPMD == 1)

#    pragma once

#    include "picongpu/simulation_defines.hpp"

#    include "picongpu/fields/Fields.hpp"
#    include "picongpu/simulation/control/MovingWindow.hpp"

#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#    include <pmacc/memory/buffers/GridBuffer.hpp>
#    include <pmacc/static_assert.hpp>

#    include <algorithm>
#    include <cstdint>
#    include <functional>
#    include <memory>
#    include <numeric>

#    include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct FromOpenPMDImpl : public T_ParamClass
        {
            //! Parameters type
            using ParamClass = T_ParamClass;

            //! Hook for boost::mpl::apply
            template<typename T_SpeciesType>
            struct apply
            {
                using type = FromOpenPMDImpl<ParamClass>;
            };

            /** Initialize the functor on the host side, includes loading the file
             *
             * This is MPI collective operation, must be called by all ranks.
             *
             * @param currentStep current time iteration
             */
            HINLINE FromOpenPMDImpl(uint32_t currentStep)
            {
                auto const& subGrid = Environment<simDim>::get().SubGrid();
                totalLocalDomainOffset = subGrid.getGlobalDomain().offset + subGrid.getLocalDomain().offset;
                loadFile();
            }

            /** Calculate the normalized density based on the file contents
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                auto const idx = totalCellOffset - totalLocalDomainOffset;
                return precisionCast<float_X>(deviceDataBox(idx).x());
            }

        private:
            //! Load density from the file to a device buffer
            void loadFile()
            {
                auto& dc = Environment<>::get().DataConnector();
                PMACC_CASSERT_MSG(_please_allocate_at_least_one_FieldTmp_in_memory_param, fieldTmpNumSlots > 0);
                auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0), true);
                auto& fieldBuffer = fieldTmp->getGridBuffer();
                // Set all to zero by default (in case the data will not be in a file)
                fieldBuffer.getHostBuffer().setValue(0.0_X);
                auto const guards = fieldBuffer.getGridLayout().getGuard();
                deviceDataBox = fieldBuffer.getDeviceBuffer().getDataBox().shift(guards);

                /* Open a series (this does not read the dataset itself).
                 * This is MPI collective and so has to be done by all ranks.
                 */
                auto& gc = Environment<simDim>::get().GridController();
                auto series = ::openPMD::Series{
                    ParamClass::filename,
                    ::openPMD::Access::READ_ONLY,
                    gc.getCommunicator().getMPIComm()};
                ::openPMD::MeshRecordComponent dataset
                    = series.iterations[ParamClass::iteration]
                          .meshes[ParamClass::datasetName][::openPMD::RecordComponent::SCALAR];
                auto const datasetExtent = dataset.getExtent();

                // Offset of the local domain in file coordinates: global coordinates, no guards, no moving window
                auto const& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                bool readFromFile = true;
                auto chunkOffset = ::openPMD::Offset(simDim, 0);
                auto chunkExtent = ::openPMD::Extent(simDim, 0);
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    chunkOffset[d] = totalLocalDomainOffset[d] - ParamClass::offset[d];
                    // Here we take care, as chunkExtent is unsigned type
                    int32_t extent = std::min(
                        static_cast<int32_t>(localDomain.size[d]),
                        static_cast<int32_t>(datasetExtent[d] - chunkOffset[d]));
                    if(extent <= 0)
                        readFromFile = false;
                    else
                        chunkExtent[d] = extent;
                }

                using ValueType = FieldTmp::ValueType::type;
                auto data = std::shared_ptr<ValueType>{nullptr};
                if(readFromFile)
                    data = dataset.loadChunk<ValueType>(chunkOffset, chunkExtent);

                // This is MPI collective and so has to be done by all ranks
                series.flush();

                auto const* rawData = data.get();
                auto const numElements
                    = std::accumulate(std::begin(chunkExtent), std::end(chunkExtent), 1u, std::multiplies<uint32_t>());
                auto hostDataBox = fieldBuffer.getHostBuffer().getDataBox().shift(guards);
                /* This loop is a bit clunky, since we are copying one simDim-dimensional array into another,
                 * and chunkExtent can be smaller than local domain size.
                 * Here we rely on the loadChunk() returning data stored in x-y-z order.
                 */
                for(uint32_t linearIdx = 0u; linearIdx < numElements; linearIdx++)
                {
                    pmacc::DataSpace<simDim> idx;
                    auto tmpLinearIdx = linearIdx;
                    for(int32_t d = simDim - 1; d >= 0; d--)
                    {
                        idx[d] = tmpLinearIdx % chunkExtent[d];
                        tmpLinearIdx /= chunkExtent[d];
                    }
                    hostDataBox(idx) = rawData[linearIdx];
                }

                // Copy host data to the device
                fieldBuffer.hostToDevice();
                __getTransactionEvent().waitForFinished();
            }

            /** Device data box with density values
             *
             * Is shifted to be indexed without guards
             */
            PMACC_ALIGN(deviceDataBox, FieldTmp::DataBoxType);

            //! Total offset of the local domain in cells
            PMACC_ALIGN(totalLocalDomainOffset, DataSpace<simDim>);
        };
    } // namespace densityProfiles
} // namespace picongpu

#endif
