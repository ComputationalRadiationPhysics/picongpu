/* Copyright 2021 Sergei Bastrakov
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
            /** Functor to check the laser compatibility to the given field solver
             *
             * Performs a run-time check and prints a warning if failed.
             *
             * @tparam T_FieldSolver field solver type
             * @tparam T_Defer technical parameter to defer evaluation;
             *                 is needed for specializations with non-template solver classes
             */
            template<typename T_FieldSolver, typename T_Defer = void>
            struct LaserChecker
            {
                /** Check the laser compatibility, prints a warning when failed.
                 *
                 * The default implementation assumes the solver is compatible to any enabled laser.
                 */
                void operator()() const
                {
                }
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
