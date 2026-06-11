/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
