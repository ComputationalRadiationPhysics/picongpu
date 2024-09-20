/* Copyright 2023 Brian Marre, Axel Huebl
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

#include <cstdint>

namespace picongpu::particles2::atomicPhysics::rateCalculation
{
    /** rate for collisional transition of ion with free electron bin
     *
     * uses second order integration(bin middle)
     *
     * @todo implement higher order integrations, Brian Marre, 2023
     *
     * @param energyElectron kinetic energy of interacting electron(/electron bin), [eV]
     * @param energyElectronBinWidth energy width of electron bin, [eV]
     * @param densityElectrons, [1/(m^3 * eV)]
     * @param sigma crossSection of collisional transition [1e6*b]
     *
     * @return unit: 1/sim.unit.time()
     */
    HDINLINE static float_X collisionalRate(
        float_X const energyElectron, // [eV]
        float_X const energyElectronBinWidth, // [eV]
        float_X const densityElectrons, // [1/(sim.unit.length()^3*eV)]
        float_X const sigma) // [1e6*b]
    {
        // constants in SI
        constexpr float_64 c_SI = picongpu::sim.si.getSpeedOfLight(); // [m/s]
        constexpr float_64 m_e_SI = picongpu::sim.si.getElectronMass(); // [kg]

        constexpr float_64 electronRestMassEnergy = m_e_SI * c_SI * c_SI / sim.si.conv.ev2Joule(1.0);
        // kg * (m^2)/(s^2) * 1/(J/eV) = Nm * eV/J = J/J * eV
        // [eV] ~ 5.11e5

        PMACC_CASSERT_MSG(
            Assumption_of_c_internal_equal_1_broken,
            (c_SI / picongpu::sim.unit.length() * picongpu::sim.unit.time() - 1.) <= 1.e-9);
        constexpr float_X c_internal = 1._X; // [sim.unit.length()/sim.unit.time()]

        constexpr float_X conversion_Factor_sigma
            = static_cast<float_X>(1.e-22 / (picongpu::sim.unit.length() * picongpu::sim.unit.length()));
        // m^2 / ((m/sim.unit.length())^2) = m^2/m^2 * sim.unit.length()^2
        // [sim.unit.length()^2] ~ 1.022e-6

        // DeltaE * sigma * rho_e/DeltaE * v_e
        return energyElectronBinWidth * sigma * conversion_Factor_sigma * densityElectrons * c_internal
            * math::sqrt(1._X - 1._X / (pmacc::math::cPow(1._X + energyElectron / electronRestMassEnergy, 2u)));
        // eV * 1e6b * sim.unit.length()^2/(1e6b) * 1/(sim.unit.length()^3*eV) * sim.unit.length()/sim.unit.time()
        //    * sqrt( unitless - [(unitless + (eV)/(eV))^2] )
        // = (eV)/(eV) * sim.unit.length()^2 *^1/sim.unit.length()^3 * sim.unit.length()/sim.unit.time()
        // = sim.unit.length()^3/sim.unit.length()^3 * 1/sim.unit.time() = 1/sim.unit.time()
        // [1/sim.unit.time()]
    }
} // namespace picongpu::particles2::atomicPhysics::rateCalculation
