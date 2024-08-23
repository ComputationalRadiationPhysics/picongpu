/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Sergei Bastrakov
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


#include "picongpu/fields/FieldJ.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/traits/GetCurrentSolver.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/tasks/FieldFactory.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/particles/memory/boxes/ParticlesBox.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>


namespace picongpu
{
    using namespace pmacc;

    template<typename A, typename B>
    using LowerMarginShapesOp =
        typename pmacc::math::CT::max<A, typename GetLowerMargin<typename GetCurrentSolver<B>::type>::type>::type;
    template<typename A, typename B>
    using UpperMarginShapesOp =
        typename pmacc::math::CT::max<A, typename GetUpperMargin<typename GetCurrentSolver<B>::type>::type>::type;

    FieldJ::FieldJ(MappingDesc const& cellDescription)
        : SimulationFieldHelper<MappingDesc>(cellDescription)
        , buffer(cellDescription.getGridLayout())
        , fieldJrecv(nullptr)
    {
        const DataSpace<simDim> coreBorderSize = cellDescription.getGridLayout().sizeWithoutGuardND();

        /* cell margins the current might spread to due to particle shapes */
        using AllSpeciesWithCurrent =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, current<>>::type;

        using LowerMarginShapes = pmacc::
            mp_fold<AllSpeciesWithCurrent, typename pmacc::math::CT::make_Int<simDim, 0>::type, LowerMarginShapesOp>;
        using UpperMarginShapes = pmacc::
            mp_fold<AllSpeciesWithCurrent, typename pmacc::math::CT::make_Int<simDim, 0>::type, UpperMarginShapesOp>;

        /* margins are always positive, also for lower margins
         * additional current interpolations and current filters on FieldJ might
         * spread the dependencies on neighboring cells
         *   -> use max(shape,filter) */
        auto const& interpolation = fields::currentInterpolation::CurrentInterpolation::get();
        auto const interpolationLowerMargin = interpolation.getLowerMargin();
        auto const interpolationUpperMargin = interpolation.getUpperMargin();
        auto const originGuard = math::max(LowerMarginShapes::toRT(), interpolationLowerMargin);
        auto const endGuard = math::max(UpperMarginShapes::toRT(), interpolationUpperMargin);

        // Type to generate a unique send tag from
        auto const sendCommTag = pmacc::traits::getUniqueId<uint32_t>();

        /*go over all directions*/
        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
        {
            DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim>(i);
            /*guarding cells depend on direction
             */
            DataSpace<simDim> guardingCells;
            for(uint32_t d = 0; d < simDim; ++d)
            {
                /*originGuard and endGuard are switch because we send data
                 * e.g. from left I get endGuardingCells and from right I originGuardingCells
                 */
                switch(relativMask[d])
                {
                    // receive from negativ side to positiv (end) guarding cells
                case -1:
                    guardingCells[d] = endGuard[d];
                    break;
                    // receive from positiv side to negativ (origin) guarding cells
                case 1:
                    guardingCells[d] = originGuard[d];
                    break;
                case 0:
                    guardingCells[d] = coreBorderSize[d];
                    break;
                };
            }
            buffer.addExchangeBuffer(i, guardingCells, sendCommTag);
        }

        /* Receive border values in own guard for "receive" communication pattern - necessary for current
         * interpolation/filter */
        const DataSpace<simDim> originRecvGuard = interpolationLowerMargin;
        const DataSpace<simDim> endRecvGuard = interpolationUpperMargin;
        if(originRecvGuard != DataSpace<simDim>::create(0) || endRecvGuard != DataSpace<simDim>::create(0))
        {
            fieldJrecv = std::make_unique<Buffer>(buffer.getDeviceBuffer(), cellDescription.getGridLayout());
            // Type to generate a unique receive tag from
            auto const recvCommTag = pmacc::traits::getUniqueId<uint32_t>();

            /*go over all directions*/
            for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
            {
                DataSpace<simDim> relativMask = Mask::getRelativeDirections<simDim>(i);
                /* guarding cells depend on direction
                 * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
                 * don't switch end and origin because this is a read buffer and no send buffer
                 */
                DataSpace<simDim> guardingCells;
                for(uint32_t d = 0; d < simDim; ++d)
                    guardingCells[d] = (relativMask[d] == -1 ? originRecvGuard[d] : endRecvGuard[d]);
                fieldJrecv->addExchange(GUARD, i, guardingCells, recvCommTag);
            }
        }
    }

    FieldJ::Buffer& FieldJ::getGridBuffer()
    {
        return buffer;
    }

    GridLayout<simDim> FieldJ::getGridLayout()
    {
        return cellDescription.getGridLayout();
    }

    EventTask FieldJ::asyncCommunication(EventTask serialEvent)
    {
        EventTask ret;
        eventSystem::startTransaction(serialEvent);
        FieldFactory::getInstance().createTaskFieldReceiveAndInsert(*this);
        ret = eventSystem::endTransaction();

        eventSystem::startTransaction(serialEvent);
        FieldFactory::getInstance().createTaskFieldSend(*this);
        ret += eventSystem::endTransaction();

        if(fieldJrecv != nullptr)
        {
            EventTask eJ = fieldJrecv->asyncCommunication(ret);
            return eJ;
        }
        else
            return ret;
    }

    void FieldJ::reset(uint32_t)
    {
    }

    void FieldJ::synchronize()
    {
        buffer.deviceToHost();
    }

    SimulationDataId FieldJ::getUniqueId()
    {
        return getName();
    }

    FieldJ::UnitValueType FieldJ::getUnit()
    {
        const float_64 unitCurrentDensity = UNIT_CHARGE / UNIT_TIME / (UNIT_LENGTH * UNIT_LENGTH);
        return UnitValueType(unitCurrentDensity, unitCurrentDensity, unitCurrentDensity);
    }

    std::vector<float_64> FieldJ::getUnitDimension()
    {
        /* L, M, T, I, theta, N, J
         *
         * J is in A/m^2
         *   -> L^-2 * I
         */
        std::vector<float_64> unitDimension(7, 0.0);
        unitDimension.at(SIBaseUnits::length) = -2.0;
        unitDimension.at(SIBaseUnits::electricCurrent) = 1.0;

        return unitDimension;
    }

    std::string FieldJ::getName()
    {
        return "J";
    }

    void FieldJ::assign(ValueType value)
    {
        buffer.getDeviceBuffer().setValue(value);
        // fieldJ.reset(false);
    }

    void FieldJ::bashField(uint32_t exchangeType)
    {
        pmacc::fields::operations::CopyGuardToExchange{}(buffer, SuperCellSize{}, exchangeType);
    }

    void FieldJ::insertField(uint32_t exchangeType)
    {
        pmacc::fields::operations::AddExchangeToBorder{}(buffer, SuperCellSize{}, exchangeType);
    }

} // namespace picongpu
