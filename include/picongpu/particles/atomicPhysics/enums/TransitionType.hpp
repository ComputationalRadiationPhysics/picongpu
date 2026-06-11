/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file transitionType enum, enum of the different sets of dataTransitions in the input data
 *
 * A dataTransition being a set of coefficients and an lower and upper state describing one or more physical
 * transitions differing in physical process and direction. For example each bound-bound transition represents, a
 * spontaneous radiative deexcitation, an electronic deexcitation and an electronic excitation.
 */

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics
{
    namespace enums
    {
        enum struct TransitionType : uint8_t
        {
            boundBound = 0u,
            boundFree = 1u,
            autonomous = 2u,
            noChange = 3u
        };
    } // namespace enums

    template<enums::TransitionType T_TransitionType>
    std::string enumToString()
    {
        if constexpr(u8(T_TransitionType) == u8(enums::TransitionType::boundBound))
            return "bound-bound";
        if constexpr(u8(T_TransitionType) == u8(enums::TransitionType::boundFree))
            return "bound-free";
        if constexpr(u8(T_TransitionType) == u8(enums::TransitionType::autonomous))
            return "autonomous";
        if constexpr(u8(T_TransitionType) == u8(enums::TransitionType::noChange))
            return "noChange";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics
