/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file ADKLaserPolarization, enum of laser polarization directions

#pragma once

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::enums
{
    enum struct ADKLaserPolarization
    {
        linearPolarization = 0,
        circularPolarization = 1
    };
} // namespace picongpu::particles::atomicPhysics::enums

namespace picongpu::particles::atomicPhysics
{
    template<enums::ADKLaserPolarization T_ADKLaserPolarization>
    std::string enumToString()
    {
        if constexpr(
            static_cast<uint8_t>(T_ADKLaserPolarization)
            == static_cast<uint8_t>(enums::ADKLaserPolarization::linearPolarization))
            return "linear polarization";
        if constexpr(
            static_cast<uint8_t>(T_ADKLaserPolarization)
            == static_cast<uint8_t>(enums::ADKLaserPolarization::circularPolarization))
            return "circular polarization";
    }
} // namespace picongpu::particles::atomicPhysics
