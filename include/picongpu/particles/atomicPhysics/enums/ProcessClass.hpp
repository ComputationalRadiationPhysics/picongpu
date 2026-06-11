/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
    std::string enumToString()
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
