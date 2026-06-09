/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

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

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::atomicPhysics
{
    //! method for calculating the FieldEnergy
    struct FieldEnergy
    {
        using VectorIdx = DataSpace<picongpu::simDim>;

        /** get field energy for the specified e Field strength
         *
         * @attention requires the **square** of the e-field strength as input!
         *
         * @param eFieldStrengthSquared, in ((unit_mass * unit_length)/(unit_time^2 * unit_charge))^2
         * @return unit: unit_energy
         */
        HDINLINE static float_X getEFieldEnergy(float_X const eFieldStrengthSquared)
        {
            /* unit: unit_charge^2 * unit_time^2 / (unit_mass * unit_length^3)
             *  * unit_length^3
             * = unit_charge^2 * unit_time^2 / unit_mass */
            constexpr float_X eps0HalfTimesCellVolume
                = (picongpu::sim.pic.getEps0() / 2._X) * picongpu::sim.pic.getCellSize().productOfComponents();

            /* unit: unit_charge^2 * unit_time^2 / unit_mass * ((unit_mass * unit_length)/(unit_time^2 *
             * unit_charge))^2 = unit_charge^2 * unit_time^2 * unit_mass^2 * unit_length^2 / (unit_mass * unit_time^4 *
             * unit_charge^2) = unit_mass * unit_length^2/ (unit_time^2 * unit_length) = unit_energy */
            return eps0HalfTimesCellVolume * eFieldStrengthSquared;
        }
    };
} // namespace picongpu::particles::atomicPhysics
