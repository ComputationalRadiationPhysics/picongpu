/* Copyright 2023 Brian Marre
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

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>

namespace picongpu::traits
{
    /** compile time functor for accessing instantiated atomicData type in
     *  numberAtomicStates flag of species
     *
     * @tparam T_IonSpecies resolved typename of species with flag
     * @returns return value contained in ::type
     */
    template<typename T_IonSpecies>
    struct GetAtomicDataType
    {
        using FrameType = typename T_IonSpecies::FrameType;

        /* throw static assert if species lacks flag */
        PMACC_CASSERT_MSG(
            Species_missing_atomicDataType_flag,
            HasFlag<FrameType, atomicDataType<>>::type::value == true);

        using AliasAtomicDataType = typename GetFlagType<FrameType, atomicDataType<>>::type;
        using type = typename pmacc::traits::Resolve<AliasAtomicDataType>::type;
    };
} // namespace picongpu::traits
