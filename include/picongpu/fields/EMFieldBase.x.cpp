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

#include "picongpu/fields/EMFieldBase.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/particles/traits/GetMarginPusher.hpp"
#include "picongpu/traits/GetMargin.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/mappings/kernel/ExchangeMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace picongpu
{
    namespace fields
    {
        template<typename A, typename B>
        using LowerMarginInterpolationOp =
            typename pmacc::math::CT::max<A, typename GetLowerMargin<typename GetInterpolation<B>::type>::type>::type;
        template<typename A, typename B>
        using UpperMarginInterpolationOp =
            typename pmacc::math::CT::max<A, typename GetUpperMargin<typename GetInterpolation<B>::type>::type>::type;
        template<typename A, typename B>
        using LowerMarginOp = typename pmacc::math::CT::max<A, typename GetLowerMarginPusher<B>::type>::type;
        template<typename A, typename B>
        using UpperMarginOp = typename pmacc::math::CT::max<A, typename GetUpperMarginPusher<B>::type>::type;

        EMFieldBase::EMFieldBase(
            MappingDesc const& cellDescription,
            pmacc::SimulationDataId const& id,
            math::Vector<int, simDim> const& lowerMargin,
            math::Vector<int, simDim> const& upperMargin)
            : SimulationFieldHelper<MappingDesc>(cellDescription)
            , id(id)
        {
            buffer = std::make_unique<Buffer>(cellDescription.getGridLayout());

            using VectorSpeciesWithInterpolation =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, interpolation<>>::type;

            using LowerMarginInterpolation = pmacc::mp_fold<
                VectorSpeciesWithInterpolation,
                typename pmacc::math::CT::make_Int<simDim, 0>::type,
                LowerMarginInterpolationOp>;
            using UpperMarginInterpolation = pmacc::mp_fold<
                VectorSpeciesWithInterpolation,
                typename pmacc::math::CT::make_Int<simDim, 0>::type,
                UpperMarginInterpolationOp>;

            /* Calculate upper and lower margin for pusher
               (currently all pusher use the interpolation of the species)
               and find maximum margin
            */
            using VectorSpeciesWithPusherAndInterpolation = typename pmacc::particles::traits::
                FilterByFlag<VectorSpeciesWithInterpolation, particlePusher<>>::type;

            using LowerMargin
                = pmacc::mp_fold<VectorSpeciesWithPusherAndInterpolation, LowerMarginInterpolation, LowerMarginOp>;
            using UpperMargin
                = pmacc::mp_fold<VectorSpeciesWithPusherAndInterpolation, UpperMarginInterpolation, UpperMarginOp>;

            auto const originGuard = pmacc::math::max(LowerMargin().toRT(), lowerMargin);
            auto const endGuard = pmacc::math::max(UpperMargin().toRT(), upperMargin);

            auto const commTag = pmacc::traits::getUniqueId<uint32_t>();

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
                buffer->addExchange(GUARD, i, guardingCells, commTag);
            }
        }

        typename EMFieldBase::Buffer& EMFieldBase::getGridBuffer()
        {
            return *buffer;
        }

        GridLayout<simDim> EMFieldBase::getGridLayout()
        {
            return cellDescription.getGridLayout();
        }

        typename EMFieldBase::DataBoxType EMFieldBase::getHostDataBox()
        {
            return buffer->getHostBuffer().getDataBox();
        }

        typename EMFieldBase::DataBoxType EMFieldBase::getDeviceDataBox()
        {
            return buffer->getDeviceBuffer().getDataBox();
        }

        EventTask EMFieldBase::asyncCommunication(EventTask serialEvent)
        {
            EventTask eB = buffer->asyncCommunication(serialEvent);
            return eB;
        }

        void EMFieldBase::reset(uint32_t)
        {
            buffer->getHostBuffer().reset(true);
            buffer->getDeviceBuffer().reset(false);
        }

        void EMFieldBase::syncToDevice()
        {
            buffer->hostToDevice();
        }

        void EMFieldBase::synchronize()
        {
            buffer->deviceToHost();
        }

        pmacc::SimulationDataId EMFieldBase::getUniqueId()
        {
            return id;
        }

    } // namespace fields
} // namespace picongpu
