/**
 * Copyright 2015 Marco Garten, Rene Widera
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

#include "simulation_defines.hpp"
#include "static_assert.hpp"
#include "traits/GetFlagType.hpp"
#include "traits/Resolve.hpp"
#include "particles/memory/frames/Frame.hpp"

namespace picongpu
{
namespace traits
{
template<typename T_Species>
struct GetIonizationEnergies
{
    typedef typename T_Species::FrameType FrameType;

    typedef typename HasFlag<FrameType, ionizationEnergies<> >::type hasIonizationEnergies;
    /* throw static assert if species has no protons or neutrons */
    PMACC_CASSERT_MSG(No_ionization_energies_are_defined_for_this_species,hasIonizationEnergies::value==true);

    typedef typename GetFlagType<FrameType,ionizationEnergies<> >::type FoundIonizationEnergiesAlias;
    /* Extract ionization energy vector from AU namespace */
    typedef typename PMacc::traits::Resolve<FoundIonizationEnergiesAlias >::type type;
};
} //namespace traits

}// namespace picongpu
