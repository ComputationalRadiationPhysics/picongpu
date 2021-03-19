/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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
#include "picongpu/fields/MaxwellSolver/None/None.def"
#include "picongpu/fields/cellType/Yee.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/types.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace none
            {
                /** Check Yee grid and time conditions
                 *
                 * This is a workaround that the condition check is only
                 * triggered if the current used solver is `NoSolver`
                 */
                template<typename T_UsedSolver, typename T_Dummy = void>
                struct ConditionCheck
                {
                };

                template<typename T_Dummy>
                struct ConditionCheck<None, T_Dummy>
                {
                    /* Courant-Friedrichs-Levy-Condition for Yee Field Solver:
                     *
                     * A workaround is to add a template dependency to the expression.
                     * `sizeof(ANY_TYPE*) != 0` is always true and defers the evaluation.
                     */
                    PMACC_CASSERT_MSG(
                        Courant_Friedrichs_Levy_condition_failure____check_your_grid_param_file,
                        (SPEED_OF_LIGHT * SPEED_OF_LIGHT * DELTA_T * DELTA_T * INV_CELL2_SUM) <= 1.0
                            && sizeof(T_Dummy*) != 0);
                };
            } // namespace none

            class None : private none::ConditionCheck<None>
            {
            private:
                typedef MappingDesc::SuperCellSize SuperCellSize;

            public:
                using CellType = cellType::Yee;

                None(MappingDesc)
                {
                }

                void update_beforeCurrent(uint32_t)
                {
                }

                void update_afterCurrent(uint32_t)
                {
                }

                static pmacc::traits::StringProperty getStringProperties()
                {
                    pmacc::traits::StringProperty propList("name", "none");
                    return propList;
                }
            };

        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Get margin for any field access in the None solver
         *
         * @tparam T_Field field type
         */
        template<typename T_Field>
        struct GetMargin<picongpu::fields::maxwellSolver::None, T_Field>
        {
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, 0>::type;
            using UpperMargin = LowerMargin;
        };
    } // namespace traits

} // namespace picongpu
