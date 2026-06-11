/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
    std::string enumToString()
    {
        if constexpr(u8(T_TransitionDirection) == u8(enums::TransitionDirection::upward))
            return "upward";
        if constexpr(u8(T_TransitionDirection) == u8(enums::TransitionDirection::downward))
            return "downward";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics
