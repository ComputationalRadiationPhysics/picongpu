/* Copyright 2020-2023 Sergei Bastrakov
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/MaxwellSolver/DispersionRelationSolver.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/Unique.hpp>

#include <cstdint>
#include <type_traits>


namespace picongpu::fields::incidentField::detail
{
    /** Calculate phase velocity for the enabled field solver and given unitless parameters
     *
     * @tparam T_Unitless unitless parameters type, must be compatible to
     * profiles::detail::BaseParamUnitless
     */
    template<typename T_Unitless>
    HINLINE float_X calculatePhaseVelocity()
    {
        auto const omega = pmacc::math::Pi<float_64>::doubleValue
            * static_cast<float_64>(sim.pic.getSpeedOfLight() / T_Unitless::WAVE_LENGTH);
        // Assume propagation along y as all laser profiles do it
        auto const direction = float3_64{T_Unitless::DIR_X, T_Unitless::DIR_Y, T_Unitless::DIR_Z};
        auto const absK = maxwellSolver::DispersionRelationSolver<Solver>{}(omega, direction);
        auto const phaseVelocity = omega / absK / sim.pic.getSpeedOfLight();
        return static_cast<float_X>(phaseVelocity);
    }

} // namespace picongpu::fields::incidentField::detail
