/*
 * SPDX-FileCopyrightText: Marco Garten, Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

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

            using FoundIonizationEnergiesAlias =
                typename pmacc::traits::GetFlagType<FrameType, ionizationEnergies<>>::type;
            /* Extract ionization energy vector from AU namespace */
            using type = typename pmacc::traits::Resolve<FoundIonizationEnergiesAlias>::type;

            static constexpr int protonNumber
                = static_cast<int>(picongpu::traits::GetAtomicNumbers<SpeciesType>::type::numberOfProtons);
            /* length of the ionization energy vector */
            static constexpr int vecLength = type::dim;
            /* assert that the number of arguments in the vector equal the proton number */
            PMACC_CASSERT_MSG(
                __The_given_number_of_ionization_energies_should_be_exactly_the_proton_number_of_the_species__,
                vecLength == protonNumber);
        };
    } // namespace traits

} // namespace picongpu
