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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

//! @file transitionType enum, enum of transition data storage groups

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics
{
    namespace enums
    {
        enum struct TransitionDirection : uint8_t
        {
            upward = 0u,
            downward = 1u,
        };
    } // namespace enums

    template<enums::TransitionDirection T_TransitionDirection>
    ALPAKA_FN_HOST std::string enumToString()
    {
        if constexpr(u8(T_TransitionDirection) == u8(enums::TransitionDirection::upward))
            return "upward";
        if constexpr(u8(T_TransitionDirection) == u8(enums::TransitionDirection::downward))
            return "downward";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics
