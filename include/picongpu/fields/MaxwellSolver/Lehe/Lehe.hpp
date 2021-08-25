/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Remi Lehe,
 *                     Sergei Bastrakov
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

#include "picongpu/fields/LaserPhysics.hpp"
#include "picongpu/fields/MaxwellSolver/LaserChecker.hpp"
#include "picongpu/fields/MaxwellSolver/Lehe/Derivative.hpp"
#include "picongpu/fields/MaxwellSolver/Lehe/Lehe.def"

#include <pmacc/traits/GetStringProperties.hpp>

#include <cstdint>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Specialization of the laser compatibility checker for for the Lehe solver
             *
             * @tparam T_CherenkovFreeDir the direction (axis) which should be free of cherenkov radiation
             */
            template<uint32_t T_cherenkovFreeDir>
            struct LaserChecker<Lehe<T_cherenkovFreeDir>>
            {
                //! This solver is not compatible to any enabled laser
                void operator()() const
                {
                    if(LaserPhysics::isEnabled())
                        log<picLog::PHYSICS>(
                            "Warning: chosen field solver is not compatible to laser\n"
                            "   The generated laser will be less accurate.\n"
                            "   For an accurate generation, either use field background or switch to Yee solver");
                }
            };
        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        template<uint32_t T_cherenkovFreeDir>
        struct StringProperties<::picongpu::fields::maxwellSolver::Lehe<T_cherenkovFreeDir>>
        {
            static StringProperty get()
            {
                auto propList = ::picongpu::fields::maxwellSolver::Lehe<T_cherenkovFreeDir>::getStringProperties();
                // overwrite the name of the solver (inherit all other properties)
                propList["name"].value = "Lehe";
                return propList;
            }
        };

    } // namespace traits
} // namespace pmacc
