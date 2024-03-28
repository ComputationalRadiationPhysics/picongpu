/* Copyright 2021-2023 Sergei Bastrakov
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
                /** Check the CFL condition
                 *
                 * @return upper bound on `c * dt` due to chosen cell size according to CFL condition
                 */
                float_X operator()() const;
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu
