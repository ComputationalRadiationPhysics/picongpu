/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera
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
#include "picongpu/fields/numericalCellTypes/NumericalCellTypes.hpp"

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

    template<typename T_CurrentInterpolation, typename T_Dummy>
    struct ConditionCheck<
        None< T_CurrentInterpolation > ,
        T_Dummy
    >
    {
        /* Courant-Friedrichs-Levy-Condition for Yee Field Solver: */
        PMACC_CASSERT_MSG(Courant_Friedrichs_Levy_condition_failure____check_your_grid_param_file,
            (SPEED_OF_LIGHT*SPEED_OF_LIGHT*DELTA_T*DELTA_T*INV_CELL2_SUM)<=1.0);
    };
} // namespace none

    template< typename T_CurrentInterpolation >
    class None : private none::ConditionCheck< None< T_CurrentInterpolation> >
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;

    public:
        using NummericalCellType = picongpu::numericalCellTypes::YeeCell;
        using CurrentInterpolation = T_CurrentInterpolation;

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
            pmacc::traits::StringProperty propList( "name", "none" );
            return propList;
        }
    };

} // namespace maxwellSolver
} // namespace fields
} // namespace picongpu
