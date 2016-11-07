/**
 * Copyright 2015-2016 Marco Garten, Rene Widera
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
struct GetEffectiveAtomicNumbers
{
    typedef typename T_Species::FrameType FrameType;

    typedef typename HasFlag<FrameType, effectiveAtomicNumbers<> >::type hasEffectiveAtomicNumbers;
    /* throw static assert if species has no predefined effective atomic numbers */
    PMACC_CASSERT_MSG(No_effective_atomic_numbers_are_defined_for_this_species,hasEffectiveAtomicNumbers::value==true);

    typedef typename GetFlagType<FrameType,effectiveAtomicNumbers<> >::type FoundEffectiveAtomicNumbersAlias;
    /* Extract vector of effective atomic numbers */
    typedef typename PMacc::traits::Resolve<FoundEffectiveAtomicNumbersAlias >::type type;
};
} //namespace traits

}// namespace picongpu
