/* Copyright 2015-2021 Marco Garten, Rene Widera
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
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/particles/memory/frames/Frame.hpp>

namespace picongpu
{
    namespace traits
    {
        template<typename T_Species>
        struct GetIonizationEnergies
        {
            using SpeciesType = T_Species;
            using FrameType = typename SpeciesType::FrameType;

            using hasIonizationEnergies = typename HasFlag<FrameType, ionizationEnergies<>>::type;
            /* throw static assert if species has no protons or neutrons */
            PMACC_CASSERT_MSG(
                No_ionization_energies_are_defined_for_this_species,
                hasIonizationEnergies::value == true);

            using FoundIonizationEnergiesAlias = typename GetFlagType<FrameType, ionizationEnergies<>>::type;
            /* Extract ionization energy vector from AU namespace */
            using type = typename pmacc::traits::Resolve<FoundIonizationEnergiesAlias>::type;

            static constexpr int protonNumber = static_cast<int>(GetAtomicNumbers<SpeciesType>::type::numberOfProtons);
            /* length of the ionization energy vector */
            static constexpr int vecLength = type::dim;
            /* assert that the number of arguments in the vector equal the proton number */
            PMACC_CASSERT_MSG(
                __The_given_number_of_ionization_energies_should_be_exactly_the_proton_number_of_the_species__,
                vecLength == protonNumber);
        };
    } // namespace traits

} // namespace picongpu
