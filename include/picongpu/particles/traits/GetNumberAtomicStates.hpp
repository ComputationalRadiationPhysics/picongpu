/* Copyright 2022-2023 Brian Marre
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

//#include "picongpu/simulation_defines.hpp"

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>

namespace picongpu::traits
{
    /** @class compile time functor for accessing data in numberAtomicStates flag of species
     *
     * @tparam T_IonSpecies resolved typename of species with flag
     * @returns return value contained in ::value
     */
    template<typename T_IonSpecies>
    struct GetNumberAtomicStates
    {
        using FrameType = typename T_IonSpecies::FrameType;

        using present = typename HasFlag<FrameType, numberAtomicStates<>>::type;
        /* throw static assert if species lacks flag */
        PMACC_CASSERT_MSG(This_species_has_no_NumberAtomicStates_Flag, present::value == true);

        using FlagTypeAlias = typename GetFlagType<FrameType, numberAtomicStates<>>::type;
        using FlagType = typename pmacc::traits::Resolve<FlagTypeAlias>::type;

        static constexpr uint16_t value = FlagType::value;
    };
} // namespace picongpu::traits
