/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file implements shorthand conversion function for enum to uint8_t
 *
 *  @attention do not use for enums with value ranges larger than uint8_t
 */

#pragma once

#include <cstdint>

namespace picongpu::particles::atomicPhysics
{
    //! static cast enum instance to uint8_t
    template<typename T_Enum>
    constexpr uint8_t u8(T_Enum const enumInstance)
    {
        return static_cast<uint8_t>(enumInstance);
    }

    //! static cast enum instance to uint32_t
    template<typename T_Enum>
    constexpr uint32_t u32(T_Enum const enumInstance)
    {
        return static_cast<uint32_t>(enumInstance);
    }

    //! static cast enum instance to bool
    template<typename T_Enum>
    constexpr bool b(T_Enum const enumInstance)
    {
        return static_cast<bool>(enumInstance);
    }
} // namespace picongpu::particles::atomicPhysics
