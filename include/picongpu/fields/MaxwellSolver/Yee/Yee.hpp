/* Copyright 2019-2022 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Sergei Bastrakov, Klaus Steiniger
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

#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/Yee/Yee.def"

#include <pmacc/traits/GetStringProperties.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Specialization of the CFL condition checker for the classic Yee solver
             *
             * @tparam T_Defer technical parameter to defer evaluation
             */
            template<typename T_Defer>
            struct CFLChecker<Yee, T_Defer>
            {
                /** Check the CFL condition, doesn't compile when failed
                 *
                 * @return value of 'X' to fulfill the condition 'c * dt <= X`
                 */
                float_X operator()() const
                {
                    // Dependance on T_Defer is required, otherwise this check would have been enforced for each setup
                    PMACC_CASSERT_MSG(
                        Courant_Friedrichs_Lewy_condition_failure____check_your_grid_param_file,
                        (SPEED_OF_LIGHT * SPEED_OF_LIGHT * DELTA_T * DELTA_T * INV_CELL2_SUM) <= 1.0
                            && sizeof(T_Defer*) != 0);

                    return 1.0_X / math::sqrt(INV_CELL2_SUM);
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
        struct StringProperties<::picongpu::fields::maxwellSolver::Yee>
        {
            static StringProperty get()
            {
                auto propList = ::picongpu::fields::maxwellSolver::Yee::getStringProperties();
                // overwrite the name of the solver (inherit all other properties)
                propList["name"].value = "Yee";
                return propList;
            }
        };

    } // namespace traits
} // namespace pmacc