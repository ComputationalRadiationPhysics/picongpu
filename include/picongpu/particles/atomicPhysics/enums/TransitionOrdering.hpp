/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file implements enum of process directions

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics
{
    namespace enums
    {
        //! predefined transitionOrderings
        enum struct TransitionOrdering : uint8_t
        {
            byLowerState = 0u,
            byUpperState = 1u
        };
    } // namespace enums

    template<enums::TransitionOrdering T_TransitionOrdering>
    std::string enumToString()
    {
        if constexpr(u8(T_TransitionOrdering) == u8(enums::TransitionOrdering::byLowerState))
            return "byLowerState";
        if constexpr(u8(T_TransitionOrdering) == u8(enums::TransitionOrdering::byUpperState))
            return "byUpperState";
    }
} // namespace picongpu::particles::atomicPhysics
