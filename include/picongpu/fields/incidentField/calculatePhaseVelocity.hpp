/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
