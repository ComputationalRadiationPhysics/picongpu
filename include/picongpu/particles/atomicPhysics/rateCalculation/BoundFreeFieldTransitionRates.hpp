/* Copyright 2023 Brian Marre, Marco Garten
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

#include "picongpu/simulation_defines.hpp" // needs ?

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics/enums/ADKLaserPolarization.hpp"
#include "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cstdint>

/** @file implements calculation of rates for bound-free field ionization atomic physics transitions
 *
 * based on the ADK ionization implementation by Marco Garten
 */

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    template<atomicPhysics::enums::ADKLaserPolarization T_ADKLaserPolarization>
    struct BoundFreeFieldTransitionRates
    {
        /** rate for bound-free field ionization transition of ion depending on the local electric field
         *
         * @tparam T_ChargeStateDataBox instantiated type of dataBox
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundFreeTransitionDataBox instantiated type of dataBox
         *
         * @param eField E-field vector, in internal units
         * @param ionizationPotentialDepression, eV
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 1/picongpu::sim.unit.time()
         */
        template<
            typename T_EFieldType,
            typename T_ChargeStateDataBox,
            typename T_AtomicStateDataBox,
            typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X rateADKFieldIonization(
            // internal units
            T_EFieldType const eFieldNorm,
            // eV
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            // eV
            float_X const ionizationEnergy = DeltaEnergyTransition::get(
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox,
                ionizationPotentialDepression,
                chargeStateDataBox);

            // get screenedCharge
            uint32_t const lowerStateClctIdx
                = boundFreeTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);
            auto const lowerStateConfigNumber = atomicStateDataBox.configNumber(lowerStateClctIdx);

            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;
            uint8_t const lowerStateChargeState = S_ConfigNumber::getChargeState(lowerStateConfigNumber);

            // e
            float_X const screenedCharge = chargeStateDataBox.screenedCharge(lowerStateChargeState) - 1._X;

            // unitless
            float_X const effectivePrincipalQuantumNumber
                = screenedCharge / math::sqrt(float_X(2.0) * ionizationEnergy);

            // electric field in atomic units
            float_X const eFieldNorm_AU = eFieldNorm / ATOMIC_UNIT_EFIELD;

            float_X const screenedChargeCubed = pmacc::math::cPow(screenedCharge, 3u);
            float_X const dBase = 4.0_X * math::exp(1._X) * screenedChargeCubed
                / (eFieldNorm_AU * pmacc::math::cPow(effectivePrincipalQuantumNumber, 4u));
            float_X const dFromADK = math::pow(dBase, effectivePrincipalQuantumNumber);

            // ionization rate (for CIRCULAR polarization)
            constexpr float_X pi = pmacc::math::Pi<float_X>::value;
            float_X const nEffCubed = pmacc::math::cPow(effectivePrincipalQuantumNumber, 3u);

            // 1/ATOMIC_UNIT_TIME
            float_X rateADK_AU = eFieldNorm_AU * pmacc::math::cPow(dFromADK, 2u) / (8._X * pi * screenedCharge)
                * math::exp(-2._X * screenedChargeCubed / (3._X * nEffCubed * eFieldNorm_AU));

            // factor from averaging over one laser cycle with LINEAR polarization
            if constexpr(
                u32(T_ADKLaserPolarization) == u32(atomicPhysics::enums::ADKLaserPolarization::linearPolarization))
                rateADK_AU *= math::sqrt(3._X * nEffCubed * eFieldNorm_AU / (pi * screenedChargeCubed));

            return rateADK_AU / ATOMIC_UNIT_TIME;
        }
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
