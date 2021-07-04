/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "picongpu/fields/FieldB.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/simulation_types.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <string>
#include <type_traits>
#include <vector>


namespace picongpu
{
    FieldB::FieldB(MappingDesc const& cellDescription)
        : SimulationFieldHelper<MappingDesc>(cellDescription)
        , id(getName())
    {
        buffer = std::make_unique<Buffer>(cellDescription.getGridLayout());
        using VectorSpeciesWithInterpolation =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, interpolation<>>::type;
        using LowerMarginInterpolation = bmpl::accumulate<
            VectorSpeciesWithInterpolation,
            typename pmacc::math::CT::make_Int<simDim, 0>::type,
            pmacc::math::CT::max<bmpl::_1, GetLowerMargin<GetInterpolation<bmpl::_2>>>>::type;
        using UpperMarginInterpolation = bmpl::accumulate<
            VectorSpeciesWithInterpolation,
            typename pmacc::math::CT::make_Int<simDim, 0>::type,
            pmacc::math::CT::max<bmpl::_1, GetUpperMargin<GetInterpolation<bmpl::_2>>>>::type;

        /* Calculate the maximum Neighbors we need from MAX(ParticleShape, FieldSolver) */
        using LowerMarginSolver = typename traits::GetLowerMargin<fields::Solver, FieldB>::type;
        using LowerMarginInterpolationAndSolver =
            typename pmacc::math::CT::max<LowerMarginInterpolation, LowerMarginSolver>::type;
        using UpperMarginSolver = typename traits::GetUpperMargin<fields::Solver, FieldB>::type;
        using UpperMarginInterpolationAndSolver =
            typename pmacc::math::CT::max<UpperMarginInterpolation, UpperMarginSolver>::type;

        /* Calculate upper and lower margin for pusher
           (currently all pusher use the interpolation of the species)
           and find maximum margin
        */
        using VectorSpeciesWithPusherAndInterpolation =
            typename pmacc::particles::traits::FilterByFlag<VectorSpeciesWithInterpolation, particlePusher<>>::type;
        using LowerMargin = typename bmpl::accumulate<
            VectorSpeciesWithPusherAndInterpolation,
            LowerMarginInterpolationAndSolver,
            pmacc::math::CT::max<bmpl::_1, GetLowerMarginPusher<bmpl::_2>>>::type;

        using UpperMargin = typename bmpl::accumulate<
            VectorSpeciesWithPusherAndInterpolation,
            UpperMarginInterpolationAndSolver,
            pmacc::math::CT::max<bmpl::_1, GetUpperMarginPusher<bmpl::_2>>>::type;

        const DataSpace<simDim> originGuard(LowerMargin().toRT());
        const DataSpace<simDim> endGuard(UpperMargin().toRT());

        /*go over all directions*/
        for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
        {
            DataSpace<simDim> relativeMask = Mask::getRelativeDirections<simDim>(i);
            /* guarding cells depend on direction
             * for negative direction use originGuard else endGuard (relative direction ZERO is ignored)
             * don't switch end and origin because this is a read buffer and no send buffer
             */
            DataSpace<simDim> guardingCells;
            for(uint32_t d = 0; d < simDim; ++d)
                guardingCells[d] = (relativeMask[d] == -1 ? originGuard[d] : endGuard[d]);
            auto const commTag = pmacc::traits::GetUniqueTypeId<FieldB>::uid();
            buffer->addExchange(GUARD, i, guardingCells, commTag);
        }
    }

    std::vector<float_64> FieldB::getUnitDimension()
    {
        /* B is in Tesla : kg / (A * s^2)
         *   -> M * T^-2 * I^-1
         */
        std::vector<float_64> unitDimension(7, 0.0);
        unitDimension.at(SIBaseUnits::mass) = 1.0;
        unitDimension.at(SIBaseUnits::time) = -2.0;
        unitDimension.at(SIBaseUnits::electricCurrent) = -1.0;
        return unitDimension;
    }

    std::string FieldB::getName()
    {
        return "B";
    }

    FieldB::Buffer& FieldB::getGridBuffer()
    {
        return *buffer;
    }

    GridLayout<simDim> FieldB::getGridLayout()
    {
        return cellDescription.getGridLayout();
    }

    FieldB::DataBoxType FieldB::getHostDataBox()
    {
        return buffer->getHostBuffer().getDataBox();
    }

    FieldB::DataBoxType FieldB::getDeviceDataBox()
    {
        return buffer->getDeviceBuffer().getDataBox();
    }

    EventTask FieldB::asyncCommunication(EventTask serialEvent)
    {
        EventTask eB = buffer->asyncCommunication(serialEvent);
        return eB;
    }

    void FieldB::reset(uint32_t)
    {
        buffer->getHostBuffer().reset(true);
        buffer->getDeviceBuffer().reset(false);
    }

    void FieldB::syncToDevice()
    {
        buffer->hostToDevice();
    }

    void FieldB::synchronize()
    {
        buffer->deviceToHost();
    }

    pmacc::SimulationDataId FieldB::getUniqueId()
    {
        return id;
    }

} // namespace picongpu
