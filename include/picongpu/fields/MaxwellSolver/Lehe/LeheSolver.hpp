/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe
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

#include "LeheSolver.def"
#include "picongpu/simulation_defines.hpp"

namespace picongpu
{
namespace leheSolver
{

} // namespace leheSolver
} // namespace picongpu

namespace pmacc
{
namespace traits
{
    template< >
    struct StringProperties< picongpu::leheSolver::LeheSolver >
    {
        static StringProperty get()
        {
            auto propList =
                ::picongpu::leheSolver::LeheSolver::getStringProperties();
            // overwrite the name of the yee solver (inherit all other properties)
            propList["name"].value = "Lehe";
            return propList;
        }
    };
} // namespace traits
} // namespace pmacc
