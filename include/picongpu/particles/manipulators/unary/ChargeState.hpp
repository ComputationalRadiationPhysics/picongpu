/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implements of functors setting the charge state of macro-particles

#pragma once

#include "picongpu/defines.hpp"

// safe to import in .param files since does not import a .param itself
#include "picongpu/particles/atomicPhysics/SetChargeState.hpp"

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

            // set to charge state
            picongpu::particles::atomicPhysics::SetChargeState{}(ion, targetNumberBoundElectrons);
        }
    };
} // namespace picongpu::particles::manipulators::unary::acc
