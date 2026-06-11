/*
 * SPDX-FileCopyrightText: Marco Garten, Rene Widera, Axel Huebl
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
        struct GetEffectiveNuclearCharge
        {
            using SpeciesType = T_Species;
            using FrameType = typename SpeciesType::FrameType;

            using hasEffectiveNuclearCharge = typename HasFlag<FrameType, effectiveNuclearCharge<>>::type;
            /* throw static assert if species has no predefined effective atomic numbers */
            PMACC_CASSERT_MSG(
                No_effective_atomic_numbers_are_defined_for_this_species,
                hasEffectiveNuclearCharge::value == true);

            using FoundEffectiveNuclearChargeAlias =
                typename pmacc::traits::GetFlagType<FrameType, effectiveNuclearCharge<>>::type;
            /* Extract vector of effective atomic numbers */
            using type = typename pmacc::traits::Resolve<FoundEffectiveNuclearChargeAlias>::type;

            static constexpr int protonNumber
                = static_cast<int>(picongpu::traits::GetAtomicNumbers<SpeciesType>::type::numberOfProtons);
            /* length of the ionization energy vector */
            static constexpr int vecLength = type::dim;
            /* assert that the number of arguments in the vector equal the proton number */
            PMACC_CASSERT_MSG(
                __The_given_number_of_effective_atomic_numbers_Z_eff_should_be_exactly_the_proton_number_of_the_species__,
                vecLength == protonNumber);
        };
    } // namespace traits
} // namespace picongpu
