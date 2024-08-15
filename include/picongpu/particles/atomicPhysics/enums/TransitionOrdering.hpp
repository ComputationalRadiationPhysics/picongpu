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

//! @file implements enum of process directions

#pragma once

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
    ALPAKA_FN_HOST std::string enumToString()
    {
        if constexpr(
            static_cast<uint8_t>(T_TransitionOrdering)
            == static_cast<uint8_t>(enums::TransitionOrdering::byLowerState))
            return "byLowerState";
        if constexpr(
            static_cast<uint8_t>(T_TransitionOrdering)
            == static_cast<uint8_t>(enums::TransitionOrdering::byUpperState))
            return "byUpperState";
    }
} // namespace picongpu::particles::atomicPhysics
