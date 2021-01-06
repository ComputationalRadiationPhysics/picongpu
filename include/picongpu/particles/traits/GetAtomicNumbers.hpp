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
        struct GetAtomicNumbers
        {
            using FrameType = typename T_Species::FrameType;

            using hasAtomicNumbers = typename HasFlag<FrameType, atomicNumbers<>>::type;
            /* throw static assert if species has no protons or neutrons */
            PMACC_CASSERT_MSG(This_species_has_no_atomic_numbers, hasAtomicNumbers::value == true);

            using FoundAtomicNumbersAlias = typename GetFlagType<FrameType, atomicNumbers<>>::type;
            using type = typename pmacc::traits::Resolve<FoundAtomicNumbersAlias>::type;
        };
    } // namespace traits

} // namespace picongpu
