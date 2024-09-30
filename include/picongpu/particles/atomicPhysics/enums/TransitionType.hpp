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
