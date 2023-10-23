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

//! @file transitionType enum, enum of transition data storage groups

#pragma once

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2
{
    namespace enums
    {
        enum struct TransitionType : uint8_t
        {
            boundBound = 0u,
            boundFree = 1u,
            autonomous = 2u
        };
    } // namespace enums

    template<enums::TransitionType T_TransitionType>
    ALPAKA_FN_HOST std::string enumToString()
    {
        if constexpr(
            static_cast<uint8_t>(T_TransitionType) == static_cast<uint8_t>(enums::TransitionType::boundBound))
            return "bound-bound";
        if constexpr(
            static_cast<uint8_t>(T_TransitionType) == static_cast<uint8_t>(enums::TransitionType::boundFree))
            return "bound-free";
        if constexpr(
            static_cast<uint8_t>(T_TransitionType) == static_cast<uint8_t>(enums::TransitionType::autonomous))
            return "autonomous";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics2

