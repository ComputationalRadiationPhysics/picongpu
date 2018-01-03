/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera
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

#include "NoSolver.def"

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"


namespace picongpu
{
    namespace noSolver
    {
        using namespace pmacc;

        /** Check Yee grid and time conditions
         *
         * This is a workaround that the condition check is only
         * triggered if the current used solver is `NoSolver`
         */
        template<typename T_UsedSolver, typename T_Dummy=void>
        struct ConditionCheck
        {
        };

        template<typename T_Dummy>
        struct ConditionCheck<NoSolver, T_Dummy>
        {
            /* Courant-Friedrichs-Levy-Condition for Yee Field Solver: */
            PMACC_CASSERT_MSG(Courant_Friedrichs_Levy_condition_failure____check_your_gridConfig_param_file,
                (SPEED_OF_LIGHT*SPEED_OF_LIGHT*DELTA_T*DELTA_T*INV_CELL2_SUM)<=1.0);
        };

        class NoSolver : private ConditionCheck<fieldSolver::FieldSolver>
        {
        private:
            typedef MappingDesc::SuperCellSize SuperCellSize;

        public:
            NoSolver(MappingDesc)
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
                pmacc::traits::StringProperty propList( "name", "none" );
                return propList;
            }
        };

    } // namespace noSolver

} // namespace picongpu
