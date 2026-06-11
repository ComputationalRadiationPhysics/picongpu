/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implements enum of different groups of physical processes

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics
{
    namespace enums
    {
        //! enum of different physics based groups of transitions
        enum struct ProcessClassGroup : uint8_t
        {
            boundBoundBased = 0u,
            boundFreeBased = 1u,
            autonomousBased = 2u,
            ionizing = 3u,
            electronicCollisional = 4u,
            electricFieldBased = 5u,
            upward = 6u,
            downward = 7u
        };
    } // namespace enums

    template<enums::ProcessClassGroup T_ProcessClassGroup>
    std::string enumToString()
    {
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::boundBoundBased))
            return "boundBound";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::boundFreeBased))
            return "boundFree";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::autonomousBased))
            return "autonomous";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::ionizing))
            return "ionizing";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::electronicCollisional))
            return "electronicCollisional";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::electricFieldBased))
            return "electricFieldBased";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::upward))
            return "upward";
        if constexpr(u8(T_ProcessClassGroup) == u8(enums::ProcessClassGroup::downward))
            return "downard";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics
