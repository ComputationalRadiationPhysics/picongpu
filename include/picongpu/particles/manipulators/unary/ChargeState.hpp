/* Copyright 2023 Brian Marre
 *
 * based on a previous implementation by Marco Garten
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

//! @file implements of functors setting the charge state of macro-particles

#pragma once

#include "picongpu/simulation_defines.hpp"

// safe to import in .param files since does not import a .param itself
#include "picongpu/particles/atomicPhysics/SetToAtomicGroundStateForChargeState.hpp"

#include <pmacc/static_assert.hpp>

#include <cstdint>

namespace picongpu::particles::manipulators::unary::acc
{
    //! see ChargeState.def for documentation
    template<uint8_t T_chargeState>
    struct ChargeState
    {
        //! set boundElectrons(charge state) of macro ion
        template<typename T_Ion>
        HDINLINE void operator()(T_Ion& ion)
        {
            constexpr float_X numberBoundElectronsNeutralAtom
                = picongpu::traits::GetAtomicNumbers<T_Ion>::type::numberOfProtons;

            // check if target charge state is physical
            PMACC_CASSERT_MSG(
                Too_high_charge_state_for_atomic_number,
                numberBoundElectronsNeutralAtom >= static_cast<float_X>(T_chargeState));

            constexpr float_X targetNumberBoundElectrons
                = numberBoundElectronsNeutralAtom - static_cast<float_X>(T_chargeState);

            // set standard charge state
            ion[boundElectrons_] = targetNumberBoundElectrons;

            // try to set atomic state
            picongpu::particles::atomicPhysics::SetToAtomicGroundStateForChargeState{}(
                ion,
                static_cast<uint8_t>(targetNumberBoundElectrons));
        }
    };
} // namespace picongpu::particles::manipulators::unary::acc
