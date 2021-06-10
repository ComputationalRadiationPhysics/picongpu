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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/Fields.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/static_assert.hpp>

#include <memory>

#include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct FromOpenPMDImpl : public T_ParamClass
        {
            using ParamClass = T_ParamClass;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = FromOpenPMDImpl<ParamClass>;
            };

            /** Initialize the functor on the host side, includes loading the file
             *
             * @param currentStep current time iteration
             */
            HINLINE FromOpenPMDImpl(uint32_t currentStep)
            {
                auto const& subGrid = Environment<simDim>::get().SubGrid();
                totalOffset = subGrid.getLocalDomain().offset;
                loadFile();
            }

            /** Calculate the normalized density based on the file contents
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                auto const localCellIdx = totalCellOffset - totalOffset;
                auto const offset = localCellIdx + SuperCellSize::toRT() * GuardSize::toRT();
                return precisionCast<float_X>(deviceDataBox(offset).x());
            }

        private:
            //! Load density from the file to a device buffer
            void loadFile()
            {
                auto& dc = Environment<>::get().DataConnector();
                PMACC_CASSERT_MSG(_please_allocate_at_least_one_FieldTmp_in_memory_param, fieldTmpNumSlots > 0);
                auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0), true);
                auto& fieldBuffer = fieldTmp->getGridBuffer();
                deviceDataBox = fieldBuffer.getDeviceBuffer().getDataBox();

                // Offset of the local domain in file coordinates: global coordinates, no guards, no moving window
                auto const& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                auto fileOffset = ::openPMD::Offset(simDim, 0);
                auto fileExtent = ::openPMD::Extent(simDim, 0);
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    fileOffset[d] = totalOffset[d];
                    fileExtent[d] = localDomain.size[d];
                }

                // Read the data from the file
                auto& gc = Environment<simDim>::get().GridController();
                auto series = ::openPMD::Series{
                    ParamClass::filename,
                    ::openPMD::Access::READ_ONLY,
                    gc.getCommunicator().getMPIComm()};
                ::openPMD::MeshRecordComponent dataset
                    = series.iterations[ParamClass::iteration]
                          .meshes[ParamClass::datasetName][::openPMD::RecordComponent::SCALAR];
                using ValueType = FieldTmp::ValueType::type;
                std::shared_ptr<ValueType> data = dataset.loadChunk<ValueType>(fileOffset, fileExtent);
                series.flush();

                auto const guards = fieldBuffer.getGridLayout().getGuard();
                auto dataBox = fieldBuffer.getHostBuffer().getDataBox().shift(guards);
                using D1Box = DataBoxDim1Access<FieldTmp::DataBoxType>;
                auto d1RAccess = D1Box{dataBox.shift(guards), localDomain.size};
                auto const* rawData = data.get();
                for(int i = 0; i < localDomain.size.productOfComponents(); ++i)
                    d1RAccess[i].x() = rawData[i];

                // Copy host data to the device
                fieldBuffer.hostToDevice();
                __getTransactionEvent().waitForFinished();
                return;
            }

            //! Device data box with density values
            PMACC_ALIGN(deviceDataBox, FieldTmp::DataBoxType);

            //! Total offset of the local domain
            PMACC_ALIGN(totalOffset, DataSpace<simDim>);
        };
    } // namespace densityProfiles
} // namespace picongpu
