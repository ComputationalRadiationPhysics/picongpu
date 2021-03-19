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

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/fields/EMFieldBase.hpp"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"
#include "picongpu/particles/traits/GetMarginPusher.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/eventSystem/EventSystem.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <boost/mpl/accumulate.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>


namespace picongpu
{
    namespace fields
    {
        template<typename T_DerivedField>
        EMFieldBase<T_DerivedField>::EMFieldBase(MappingDesc const& cellDescription, pmacc::SimulationDataId const& id)
            : SimulationFieldHelper<MappingDesc>(cellDescription)
            , id(id)
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
            using LowerMarginSolver = typename traits::GetMargin<fields::Solver, DerivedField>::LowerMargin;
            using LowerMarginInterpolationAndSolver =
                typename pmacc::math::CT::max<LowerMarginInterpolation, LowerMarginSolver>::type;
            using UpperMarginSolver = typename traits::GetMargin<fields::Solver, DerivedField>::UpperMargin;
            using UpperMarginInterpolationAndSolver =
                typename pmacc::math::CT::max<UpperMarginInterpolation, UpperMarginSolver>::type;

            /* Calculate upper and lower margin for pusher
               (currently all pusher use the interpolation of the species)
               and find maximum margin
            */
            using VectorSpeciesWithPusherAndInterpolation = typename pmacc::particles::traits::
                FilterByFlag<VectorSpeciesWithInterpolation, particlePusher<>>::type;
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
                auto const commTag = pmacc::traits::GetUniqueTypeId<DerivedField>::uid();
                buffer->addExchange(GUARD, i, guardingCells, commTag);
            }
        }

        template<typename T_DerivedField>
        typename EMFieldBase<T_DerivedField>::Buffer& EMFieldBase<T_DerivedField>::getGridBuffer()
        {
            return *buffer;
        }

        template<typename T_DerivedField>
        GridLayout<simDim> EMFieldBase<T_DerivedField>::getGridLayout()
        {
            return cellDescription.getGridLayout();
        }

        template<typename T_DerivedField>
        typename EMFieldBase<T_DerivedField>::DataBoxType EMFieldBase<T_DerivedField>::getHostDataBox()
        {
            return buffer->getHostBuffer().getDataBox();
        }

        template<typename T_DerivedField>
        typename EMFieldBase<T_DerivedField>::DataBoxType EMFieldBase<T_DerivedField>::getDeviceDataBox()
        {
            return buffer->getDeviceBuffer().getDataBox();
        }

        template<typename T_DerivedField>
        EventTask EMFieldBase<T_DerivedField>::asyncCommunication(EventTask serialEvent)
        {
            EventTask eB = buffer->asyncCommunication(serialEvent);
            return eB;
        }

        template<typename T_DerivedField>
        void EMFieldBase<T_DerivedField>::reset(uint32_t)
        {
            buffer->getHostBuffer().reset(true);
            buffer->getDeviceBuffer().reset(false);
        }

        template<typename T_DerivedField>
        void EMFieldBase<T_DerivedField>::syncToDevice()
        {
            buffer->hostToDevice();
        }

        template<typename T_DerivedField>
        void EMFieldBase<T_DerivedField>::synchronize()
        {
            buffer->deviceToHost();
        }

        template<typename T_DerivedField>
        pmacc::SimulationDataId EMFieldBase<T_DerivedField>::getUniqueId()
        {
            return id;
        }

    } // namespace fields
} // namespace picongpu
