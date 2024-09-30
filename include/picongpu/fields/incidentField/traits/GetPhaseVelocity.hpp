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
#include "picongpu/fields/incidentField/traits/GetFunctor.hpp"
#include "picongpu/fields/incidentField/traits/GetPhaseVelocity.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/Unique.hpp>

#include <cstdint>
#include <type_traits>


namespace picongpu::fields::incidentField::traits
{
    namespace detail
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

        /** Get phase velocity for the enabled field solver and given incident field profile
         *
         * General implementation for parametrized profiles with parameters compatible to profiles::BaseParam
         *
         * @tparam T_Profile profile type
         */
        template<typename T_Profile>
        struct GetPhaseVelocity
        {
            HINLINE float_X operator()() const
            {
                using Functor = FunctorIncidentE<T_Profile>;
                using Unitless = typename Functor::Unitless;
                return calculatePhaseVelocity<Unitless>();
            }
        };
    } // namespace detail

    /** Get phase velocity for the enabled field solver and given incident field profile
     *
     * @tparam T_Profile profile type
     */
    template<typename T_Profile>
    HINLINE float_X getPhaseVelocity()
    {
        return detail::GetPhaseVelocity<T_Profile>{}();
    }

} // namespace picongpu::fields::incidentField::traits
