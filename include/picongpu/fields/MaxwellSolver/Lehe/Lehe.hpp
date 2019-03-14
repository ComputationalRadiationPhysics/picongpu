/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe
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

#include "picongpu/fields/MaxwellSolver/Lehe/Lehe.def"
#include "picongpu/fields/MaxwellSolver/Lehe/Curl.hpp"
#include "picongpu/simulation_defines.hpp"

namespace pmacc
{
namespace traits
{
    template<
        typename T_CurrentInterpolation,
        typename T_CherenkovFreeDir
    >
    struct StringProperties<
        ::picongpu::fields::maxwellSolver::Lehe<
            T_CurrentInterpolation,
            T_CherenkovFreeDir
        >
    >
    {
        static StringProperty get()
        {
            auto propList =
                ::picongpu::fields::maxwellSolver::Lehe<
                    T_CurrentInterpolation,
                    T_CherenkovFreeDir
                >::getStringProperties();
            // overwrite the name of the yee solver (inherit all other properties)
            propList["name"].value = "Lehe";
            return propList;
        }
    };
} // namespace traits
} // namespace pmacc
