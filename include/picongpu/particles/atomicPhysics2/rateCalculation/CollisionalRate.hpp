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

namespace picongpu::particles2::atomicPhysics2::rateCalculation
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
     * @return unit: 1/UNIT_TIME
     */
    HDINLINE static float_X collisionalRate(
        float_X const energyElectron, // [eV]
        float_X const energyElectronBinWidth, // [eV]
        float_X const densityElectrons, // [1/(UNIT_LENGTH^3*eV)]
        float_X const sigma) // [1e6*b]
    {
        // constants in SI
        constexpr float_64 c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // [m/s]
        constexpr float_64 m_e_SI = picongpu::SI::ELECTRON_MASS_SI; // [kg]

        constexpr float_64 electronRestMassEnergy = m_e_SI * c_SI * c_SI / picongpu::UNITCONV_eV_to_Joule;
        // kg * (m^2)/(s^2) * 1/(J/eV) = Nm * eV/J = J/J * eV
        // [eV] ~ 5.11e5

        PMACC_CASSERT_MSG(
            Assumption_of_c_internal_equal_1_broken,
            (c_SI / picongpu::UNIT_LENGTH * picongpu::UNIT_TIME - 1.) <= 1.e-9);
        constexpr float_X c_internal = 1._X; // [UNIT_LENGTH/UNIT_TIME]

        constexpr float_X conversion_Factor_sigma
            = static_cast<float_X>(1.e-22 / (picongpu::UNIT_LENGTH * picongpu::UNIT_LENGTH));
        // m^2 / ((m/UNIT_LENGTH)^2) = m^2/m^2 * UNIT_LENGTH^2
        // [UNIT_LENGTH^2] ~ 1.022e-6

        // DeltaE * sigma * rho_e/DeltaE * v_e
        return energyElectronBinWidth * sigma * conversion_Factor_sigma * densityElectrons * c_internal
            * math::sqrt(1._X - 1._X / (pmacc::math::cPow(1._X + energyElectron / electronRestMassEnergy, 2u)));
        // eV * 1e6b * UNIT_LENGTH^2/(1e6b) * 1/(UNIT_LENGTH^3*eV) * UNIT_LENGTH/UNIT_TIME
        //    * sqrt( unitless - [(unitless + (eV)/(eV))^2] )
        // = (eV)/(eV) * UNIT_LENGTH^2 *^1/UNIT_LENGTH^3 * UNIT_LENGTH/UNIT_TIME
        // = UNIT_LENGTH^3/UNIT_LENGTH^3 * 1/UNIT_TIME = 1/UNIT_TIME
        // [1/UNIT_TIME]
    }
} // namespace picongpu::particles2::atomicPhysics2::rateCalculation
