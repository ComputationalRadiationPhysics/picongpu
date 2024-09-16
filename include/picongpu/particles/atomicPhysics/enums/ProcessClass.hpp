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

//! @file processClass enum, enum of physical processes

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics
{
    namespace enums
    {
        enum struct ProcessClass : uint8_t
        {
            noChange = 0u,
            spontaneousDeexcitation = 1u,
            electronicExcitation = 2u,
            electronicDeexcitation = 3u,
            electronicIonization = 4u,
            autonomousIonization = 5u,
            fieldIonization = 6u,
            ipdIonization = 7u,
        };
    } // namespace enums

    template<enums::ProcessClass T_ProcessClass>
    ALPAKA_FN_HOST std::string enumToString()
    {
        if constexpr(u8(T_ProcessClass) == u8(enums::ProcessClass::noChange))
            return "noChange";
        if constexpr(u8(T_ProcessClass) == u8(enums::ProcessClass::spontaneousDeexcitation))
            return "spontaneousDeexcitation";
        if constexpr(u8(T_ProcessClass) == u8(enums::ProcessClass::electronicExcitation))
            return "electronicExcitation";
        if constexpr(u8(T_ProcessClass) == u8(enums::ProcessClass::electronicDeexcitation))
            return "electronicDeexcitation";
        if constexpr(u8(T_ProcessClass) == u8(enums::ProcessClass::electronicIonization))
            return "electronicIonization";
        if constexpr(u8(T_ProcessClass) == u8(enums::ProcessClass::autonomousIonization))
            return "autonomousIonization";
        if constexpr(u8(T_ProcessClass) == u8(enums::ProcessClass::fieldIonization))
            return "fieldIonization";
        if constexpr(u8(T_ProcessClass) == u8(enums::ProcessClass::ipdIonization))
            return "ipdIonization";
        return "unknown";
    }
} // namespace picongpu::particles::atomicPhysics
