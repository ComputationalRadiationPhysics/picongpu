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

//! @file implements enum of different groups of process classes

#pragma once

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2
{
    namespace processClass
    {
        //! predefined groups of processClass
        enum struct ProcessClassGroup : uint8_t
        {
            boundBoundBased = 0u,
            boundFreeBased = 1u,
            autonomousBased = 2u,
            ionizing = 3u,
            electronicCollisional = 4u,
            upward = 5u,
            downward = 6u
        };
    } // namespace processClass

    template<processClass::ProcessClassGroup T_ProcessClassGroup>
    ALPAKA_FN_HOST std::string enumToString()
    {
        if constexpr(
            static_cast<uint8_t>(T_ProcessClassGroup)
            == static_cast<uint8_t>(processClass::ProcessClassGroup::boundBoundBased))
            return "boundBound";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClassGroup)
            == static_cast<uint8_t>(processClass::ProcessClassGroup::boundFreeBased))
            return "boundFree";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClassGroup)
            == static_cast<uint8_t>(processClass::ProcessClassGroup::autonomousBased))
            return "autonomous";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClassGroup)
            == static_cast<uint8_t>(processClass::ProcessClassGroup::ionizing))
            return "ionizing";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClassGroup)
            == static_cast<uint8_t>(processClass::ProcessClassGroup::electronicCollisional))
            return "electronicCollisional";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClassGroup) == static_cast<uint8_t>(processClass::ProcessClassGroup::upward))
            return "upward";
        if constexpr(
            static_cast<uint8_t>(T_ProcessClassGroup)
            == static_cast<uint8_t>(processClass::ProcessClassGroup::downward))
            return "downard";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics2
