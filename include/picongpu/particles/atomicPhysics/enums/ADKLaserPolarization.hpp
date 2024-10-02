/* Copyright 2024 Brian Marre
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
