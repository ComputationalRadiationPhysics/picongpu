/* Copyright 2021-2022 Sergei Bastrakov
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


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Functor to check the Courant-Friedrichs-Lewy-Condition for the given field solver
             *
             * Performs either a compile-time check or a run-time check and throws if failed.
             *
             * @tparam T_FieldSolver field solver type
             * @tparam T_Defer technical parameter to defer evaluation;
             *                 is needed for specializations with non-template solver classes
             */
            template<typename T_FieldSolver, typename T_Defer = void>
            struct CFLChecker
            {
                /** Check the CFL condition, doesn't compile when failed
                 *
                 * The default implementation checks the basic explicit PIC assumption c * dt <= min(dx, dy, dz).
                 * Note that generally FDTD-type field solvers have a more strict condition, and have to provide a
                 * specialization.
                 *
                 * @return value of 'X' to fulfill the condition 'c * dt <= X`
                 */
                float_X operator()() const
                {
                    // For 2d the value of dz does not matter for this check.
                    // Using dx instead is fine here since we are taking min grid step.
                    constexpr auto usedDz = (simDim == 3) ? CELL_DEPTH : CELL_WIDTH;
                    constexpr auto minCellSize = std::min({CELL_WIDTH, CELL_HEIGHT, usedDz});
                    /* Dependance on T_Defer is required, otherwise this check would have been enforced for each setup
                     * (in this case, could have depended on T_FieldSolver as well)
                     */
                    PMACC_CASSERT_MSG(
                        Courant_Friedrichs_Lewy_condition_failure____check_your_grid_param_file,
                        (SPEED_OF_LIGHT * DELTA_T / minCellSize <= 1.0) && sizeof(T_Defer*) != 0);

                    return minCellSize;
                }
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
