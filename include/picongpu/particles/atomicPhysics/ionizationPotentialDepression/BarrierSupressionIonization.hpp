/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2024 Brian Marre
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

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    struct BarrierSupressionIonization
    {
        /** get ionization potential depression(IPD) due to electric field according to the barrier suppression
         *  ionization model
         *
         * @param screenedCharge, in e
         * @param electricFieldNormAU, in sim.atomicUnit.eField()
         *
         * @return unit: eV
         */
        HDINLINE static float_X getIPD(float_X const screenedCharge, float_X const electricFieldNormAU)
        {
            // Hartree = sim.atomicUnit.energy()
            return picongpu::sim.si.conv().auEnergy2eV(2._X * math::sqrt(screenedCharge * electricFieldNormAU));
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
