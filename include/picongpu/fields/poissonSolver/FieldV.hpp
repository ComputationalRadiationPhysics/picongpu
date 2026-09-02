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
#include "picongpu/fields/YeeCell.hpp"
#include "picongpu/traits/FieldPosition.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/memory/buffers/GridBuffer.hpp>

namespace picongpu::fields::poissonSolver
{
    struct FieldV : pmacc::ISimulationData
    {
        using ValueType = float_64;

        std::shared_ptr<GridBuffer<ValueType, simDim>> fieldVBuffer;

        //! Unit type of field components
        using UnitValueType = float1_64;

        using DataBoxType = pmacc::DataBox<PitchedBox<ValueType, simDim>>;

        //! Number of components of ValueType, for serialization
        static constexpr int numComponents = 1u;

        FieldV(picongpu::MappingDesc const mappingDesc)
            : fieldVBuffer{std::make_shared<GridBuffer<float_64, simDim>>(mappingDesc.getGridLayout())}
        {
        }

        void synchronize() override
        {
            fieldVBuffer->getHostBuffer().copyFrom(fieldVBuffer->getDeviceBuffer());
        }

        //! Get the host data box for the field values
        DataBoxType getHostDataBox()
        {
            return fieldVBuffer->getHostBuffer().getDataBox();
        }

        //! Get the device data box for the field values
        DataBoxType getDeviceDataBox()
        {
            return fieldVBuffer->getDeviceBuffer().getDataBox();
        }

        GridLayout<simDim> getGridLayout()
        {
            return fieldVBuffer->getGridLayout();
        }

        /**
         * Return the globally unique identifier for this simulation data.
         *
         * @return globally unique identifier
         */
        SimulationDataId getUniqueId() override
        {
            return "FieldV";
        }

        static std::string getName()
        {
            return "FieldV";
        }

        static UnitValueType getUnit()
        {
            return UnitValueType{sim.unit.eField() * sim.unit.length()};
        }

        static std::vector<float_64> getUnitDimension()
        {
            /* V is in volts: V  = kg * m^2 / (A * s^3)
             *   -> L^2 * M * T^-3 * I^-1
             */
            std::vector<float_64> unitDimension(7, 0.0);
            unitDimension.at(SIBaseUnits::length) = 2.0;
            unitDimension.at(SIBaseUnits::mass) = 1.0;
            unitDimension.at(SIBaseUnits::time) = -3.0;
            unitDimension.at(SIBaseUnits::electricCurrent) = -1.0;
            return unitDimension;
        }
    };
} // namespace picongpu::fields::poissonSolver

namespace picongpu::traits
{
    template<>
    struct FieldPosition<fields::YeeCell, fields::poissonSolver::FieldV, simDim>
    {
        using VectorVectorDD = ::pmacc::math::Vector<floatD_X, simDim> const;

        HDINLINE FieldPosition() = default;

        HDINLINE VectorVectorDD operator()() const
        {
            return VectorVectorDD::create(floatD_X::create(0.0));
        }
    };
} // namespace picongpu::traits
