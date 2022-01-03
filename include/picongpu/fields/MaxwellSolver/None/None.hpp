/* Copyright 2013-2022 Axel Huebl, Heiko Burau, Rene Widera
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

#include "picongpu/fields/MaxwellSolver/LaserChecker.hpp"
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
            class None
            {
            private:
                using SuperCellSize = MappingDesc::SuperCellSize;

            public:
                using CellType = cellType::Yee;

                None(MappingDesc)
                {
                    LaserChecker<None>{}();
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
