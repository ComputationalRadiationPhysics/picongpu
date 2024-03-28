/* Copyright 2014-2023 Marco Garten, Rene Widera
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

#include "picongpu/simulation_defines.hpp"

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
    HDINLINE float_X getChargeState(const T_Particle& particle)
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
        const float_X protonNumber = GetAtomicNumbers<T_Particle>::type::numberOfProtons;
        return protonNumber - particle[boundElectrons_];
    }
} // namespace picongpu::traits::attribute
