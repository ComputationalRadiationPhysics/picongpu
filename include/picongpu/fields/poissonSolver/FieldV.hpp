/* Copyright 2025 Tapish Narwal, Luca Pennati, Rene Widera
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

#include "picongpu/defines.hpp"

#include <pmacc/memory/buffers/GridBuffer.hpp>

namespace picongpu::fields::poissonSolver
{
    struct FieldV : pmacc::ISimulationData
    {
        std::shared_ptr<GridBuffer<float_64, simDim>> fieldVBuffer;

        FieldV(picongpu::MappingDesc const mappingDesc)
            : fieldVBuffer{std::make_shared<GridBuffer<float_64, simDim>>(mappingDesc.getGridLayout())}
        {
        }

        void synchronize() override
        {
            fieldVBuffer->getDeviceBuffer().copyFrom(fieldVBuffer->getHostBuffer());
        };

        /**
         * Return the globally unique identifier for this simulation data.
         *
         * @return globally unique identifier
         */
        SimulationDataId getUniqueId() override
        {
            return "FieldV";
        };
    };
} // namespace picongpu::fields::poissonSolver
