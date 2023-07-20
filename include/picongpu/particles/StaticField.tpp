/* Copyright 2023 Rene Widera
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/simulation/stage/CurrentBackground.hpp"
#include "picongpu/simulation/stage/CurrentDeposition.hpp"
#include "picongpu/simulation/stage/CurrentInterpolationAndAdditionToEMF.hpp"
#include "picongpu/simulation/stage/CurrentReset.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>

#include <cstdint>


namespace picongpu::particles
{
    void StaticField::operator()(uint32_t currentStep, uint32_t numFildSolverSteps)
    {
        std::cout << "start StaticField " << currentStep << " for " << numFildSolverSteps << " iterations."
                  << std::endl;
        simulation::stage::CurrentReset{}(currentStep);
        simulation::stage::CurrentDeposition{}(currentStep);

        DataConnector& dc = Environment<>::get().DataConnector();

        auto currentBackground
            = dc.get<simulation::stage::CurrentBackground>(simulation::stage::CurrentBackground::getName());
        (*currentBackground)(currentStep);

        auto fieldSolver = dc.get<fields::Solver>(fields::Solver::getName());
        auto currentInterpolationAndAdditionToEMF = dc.get<simulation::stage::CurrentInterpolationAndAdditionToEMF>(
            simulation::stage::CurrentInterpolationAndAdditionToEMF::getName());

        for(uint32_t step = 0u; step < numFildSolverSteps; ++step)
        {
            fieldSolver->update_beforeCurrent(currentStep);
            (*currentInterpolationAndAdditionToEMF)(currentStep, *fieldSolver, step == 0);
            fieldSolver->update_afterCurrent(currentStep);
        }

        simulation::stage::CurrentReset{}(currentStep);
        std::cout << "end StaticField" << std::endl;
    }
} // namespace picongpu::particles
