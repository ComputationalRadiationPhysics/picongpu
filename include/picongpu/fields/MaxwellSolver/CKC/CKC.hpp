/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe,
 *                     Sergei Bastrakov, Lennert Sprenger
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

#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/CKC/CKC.def"
#include "picongpu/fields/MaxwellSolver/CKC/Derivative.hpp"
#include "picongpu/fields/MaxwellSolver/DispersionRelation.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/traits/GetStringProperties.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Specialization of the CFL condition checker for CKC solver
             *
             * @tparam T_Defer technical parameter to defer evaluation
             */
            template<typename T_Defer>
            struct CFLChecker<CKC, T_Defer>
            {
                /** Check the CFL condition according to the paper, doesn't compile when failed
                 *
                 * @return upper bound on `c * dt` due to chosen cell size according to CFL condition
                 */
                float_X operator()() const
                {
                    // cellSize is not constexpr currently, so make an own constexpr array
                    constexpr float_X step[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                    constexpr float_X cdt = SPEED_OF_LIGHT * getTimeStep(); // c * dt

                    constexpr float_64 delta = std::min({step[0], step[1], step[2]});

                    // Dependence on T_Defer is required, otherwise this check would have been enforced for each setup
                    PMACC_CASSERT_MSG(
                        Courant_Friedrichs_Lewy_condition_failure____check_your_grid_param_file,
                        (cdt <= delta) && sizeof(T_Defer*) != 0);

                    return delta;
                }
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        template<>
        struct StringProperties<::picongpu::fields::maxwellSolver::CKC>
        {
            static StringProperty get()
            {
                auto propList = ::picongpu::fields::maxwellSolver::CKC::getStringProperties();
                // overwrite the name of the solver (inherit all other properties)
                propList["name"].value = "CK";
                return propList;
            }
        };

    } // namespace traits
} // namespace pmacc
