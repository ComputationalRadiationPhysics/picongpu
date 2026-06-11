/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implements setter for charge state

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"

#include <pmacc/assert.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics
{
    struct SetChargeState
    {
        //! @attention invalidates the atomicStateCollectionIndex attribute of macro ions
        template<typename T_Ion>
        DINLINE void operator()(T_Ion& ion, float_X numberBoundElectrons)
        {
            PMACC_DEVICE_ASSERT_MSG(numberBoundElectrons >= 0._X, "Number of bound electrons must be >= 0");
            PMACC_DEVICE_ASSERT_MSG(
                numberBoundElectrons <= picongpu::traits::GetAtomicNumbers<T_Ion>::type::numberOfProtons,
                "Number of bound electrons must be <= numberOfProtons species");

            ion[boundElectrons_] = numberBoundElectrons;

            if constexpr(traits::hasParticleTypeTag<Tags::Ion, T_Ion>())
            {
                /* both boundElectrons and atomicStateCollectionIndex particle attribute must be set consistently,
                 *  but we lack access to the atomicStateData to correctly update atomicStateCollectionIndex
                 *
                 * Instead we invalidate it by purpose and check at the start of the atomicPhysics step for
                 * consistency and set all inconsistent macro-ions to their respective atomic ground state.
                 */

                // invalidate atomicStateCollectionIndex particle attribute for easier detection
                ion[atomicStateCollectionIndex_] = std::numeric_limits<uint32_t>::max();
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics
