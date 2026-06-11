/*
 * SPDX-FileCopyrightText: Marco Garten, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"

#include <pmacc/algorithms/TypeCast.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu::traits::attribute
{
    /** get the charge state of a macro particle
     *
     * This function trait considers the `boundElectrons` attribute if it is set.
     * Charge states do not add up and also the various particles in a macro particle
     * do NOT have different charge states where one would average over them.
     *
     * @param particle a reference to a particle
     * @return charge of the macro particle
     */
    template<typename T_Particle>
    HDINLINE float_X getChargeState(T_Particle const& particle)
    {
        constexpr bool hasBoundElectrons = pmacc::traits::HasIdentifier<T_Particle, boundElectrons>::type::value;
        PMACC_CASSERT_MSG(
            This_species_has_only_one_charge_state_add_species_attribute_boundElectrons,
            hasBoundElectrons);

        using HasAtomicNumbers = typename pmacc::traits::HasFlag<T_Particle, atomicNumbers<>>::type;
        PMACC_CASSERT_MSG_TYPE(
            Having_boundElectrons_particle_attribute_requires_atomicNumbers_flag,
            T_Particle,
            HasAtomicNumbers::value);
        float_X const protonNumber = picongpu::traits::GetAtomicNumbers<T_Particle>::type::numberOfProtons;
        return protonNumber - particle[boundElectrons_];
    }
} // namespace picongpu::traits::attribute
