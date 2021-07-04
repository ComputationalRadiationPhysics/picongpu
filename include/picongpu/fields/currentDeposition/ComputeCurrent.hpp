/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Sergei Bastrakov
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

#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/currentDeposition/ComputeCurrent.kernel"


namespace picongpu
{
    namespace fields
    {
        namespace currentDeposition
        {
            /** Compute current density created by a species in an area
             *
             * @tparam T_area area to compute currents in
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param fieldJ current density field instance
             * @param currentStep index of time iteration
             */
            template<uint32_t T_area, class T_Species>
            HINLINE void computeCurrent(T_Species& species, FieldJ& fieldJ, uint32_t currentStep)
            {
                using FrameType = typename T_Species::FrameType;
                typedef typename pmacc::traits::Resolve<typename GetFlagType<FrameType, current<>>::type>::type
                    ParticleCurrentSolver;

                using FrameSolver
                    = currentSolver::ComputePerFrame<ParticleCurrentSolver, Velocity, MappingDesc::SuperCellSize>;

                typedef SuperCellDescription<
                    typename MappingDesc::SuperCellSize,
                    typename GetMargin<ParticleCurrentSolver>::LowerMargin,
                    typename GetMargin<ParticleCurrentSolver>::UpperMargin>
                    BlockArea;

                using Strategy = currentSolver::traits::GetStrategy_t<FrameSolver>;

                constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                    pmacc::math::CT::volume<SuperCellSize>::type::value * Strategy::workerMultiplier>::value;

                auto const depositionKernel = currentSolver::KernelComputeCurrent<numWorkers, BlockArea>{};

                typename T_Species::ParticlesBoxType pBox = species.getDeviceParticlesBox();
                FieldJ::DataBoxType jBox = fieldJ.getGridBuffer().getDeviceBuffer().getDataBox();
                FrameSolver solver(DELTA_T);

                auto const deposit = currentSolver::Deposit<Strategy>{};
                deposit.template execute<T_area, numWorkers>(cellDescription, depositionKernel, solver, jBox, pBox);
            }

        } // namespace currentDeposition
    } // namespace fields
} // namespace picongpu
