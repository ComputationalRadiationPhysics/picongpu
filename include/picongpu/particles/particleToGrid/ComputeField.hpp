/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Pawel Ordyna, Sergei Bastrakov
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

#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/particles/particleToGrid/ComputeField.kernel"
#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/eventSystem/EventSystem.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/tasks/FieldFactory.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <boost/mpl/accumulate.hpp>

#include <memory>
#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            // Used to be FieldTmp::computeValue
            /** Compute current density created by a species in an area
             *
             * @tparam T_area area to compute currents in
             * @tparam T_Species particle species type
             * @tparam Filter particle filter used to filter contributing particles
             *         (default is all particles contribute)
             *
             * @param species particle species
             * @param currentStep index of time iteration
             */
            template<uint32_t AREA, class FrameSolver, typename Filter = particles::filter::All, class ParticlesClass>
            HINLINE void computeValue(ParticlesClass& parClass, uint32_t currentStep, FieldTmp& result)
            {
                typedef SuperCellDescription<
                    typename MappingDesc::SuperCellSize,
                    typename FrameSolver::LowerMargin,
                    typename FrameSolver::UpperMargin>
                    BlockArea;

                auto const cellDescription = result.getCellDescription();
                StrideMapping<AREA, 3, MappingDesc> mapper(cellDescription);
                typename ParticlesClass::ParticlesBoxType pBox = parClass.getDeviceParticlesBox();
                FieldTmp::DataBoxType tmpBox = result.getDeviceDataBox();
                FrameSolver solver;
                using ParticleFilter = typename Filter ::template apply<ParticlesClass>::type;
                ParticleFilter particleFilter;
                constexpr uint32_t numWorkers
                    = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                do
                {
                    PMACC_KERNEL(KernelComputeSupercells<numWorkers, BlockArea>{})
                    (mapper.getGridDim(), numWorkers)(tmpBox, pBox, solver, particleFilter, mapper);
                } while(mapper.next());
            }

        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
