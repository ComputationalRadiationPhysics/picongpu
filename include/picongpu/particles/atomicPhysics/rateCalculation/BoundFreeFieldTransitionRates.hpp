/* Copyright 2024-2025 Brian Marre, Marco Garten
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
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics/FieldEnergy.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"
#include "picongpu/particles/atomicPhysics/enums/ADKLaserPolarization.hpp"
#include "picongpu/particles/atomicPhysics/stateRepresentation/ConfigNumber.hpp"

#include <pmacc/algorithms/math.hpp>
#include <pmacc/types.hpp>

#include <cstdint>
#include <limits>

/** @file implements calculation of rates for bound-free field ionization atomic physics transitions
 *
 * based on the ADK ionization implementation by Marco Garten
 */

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    /** methods for calculating bound-free field based transition rates
     *
     * @tparam T_ADKLaserPolarization, laser polarization type to use in the ADK model
     */
    template<atomicPhysics::enums::ADKLaserPolarization T_ADKLaserPolarization>
    struct BoundFreeFieldTransitionRates
    {
    private:
        //! input required for calling rate formula
        template<typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        struct RateFormulaVariables
        {
            // unitless
            PMACC_ALIGN(nEff, float_X);
            // in elementary charge
            PMACC_ALIGN(Z, float_X);

            /** initialize
             *
             * @param ionizationEnergy, in eV
             * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
             * @param atomicStateDataBox access to atomic state property data
             * @param boundBoundTransitionDataBox access to bound-bound transition data
             *
             * @attention ionizationEnergy must be > 0
             */
            HDINLINE RateFormulaVariables(
                float_X const ionizationEnergy,
                uint32_t const transitionCollectionIndex,
                T_AtomicStateDataBox const atomicStateDataBox,
                T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
            {
                Z = BoundFreeFieldTransitionRates::screenedCharge(
                    transitionCollectionIndex,
                    atomicStateDataBox,
                    boundFreeTransitionDataBox);

                nEff = BoundFreeFieldTransitionRates::effectivePrincipalQuantumNumber(Z, ionizationEnergy);
            }
        };

    public:
        /** get effective principal quantum number
         *
         * @param screenedCharge, in e
         * @param ionizationEnergy, in eV
         *
         * @attention ionizationEnergy must be > 0
         *
         * @return unitless
         */
        HDINLINE static float_X effectivePrincipalQuantumNumber(
            float_X const screenedCharge,
            float_X const ionizationEnergy)
        {
            return screenedCharge / math::sqrt(2._X * picongpu::sim.si.conv().eV2auEnergy(ionizationEnergy));
        }

        /** get screened charge for ionization electron
         *
         * @return unit: e
         */
        template<typename T_AtomicStateDataBox, typename T_BoundFreeTransitionDataBox>
        HDINLINE static float_X screenedCharge(
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            uint32_t const lowerStateClctIdx
                = boundFreeTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);
            return atomicStateDataBox.screenedCharge(lowerStateClctIdx);
        }

        /** actual rate rateFormula
         *
         * @tparam T_ReturnType type and precision of return, usually float_32 or float_64
         *
         * @param Z screenedCharge for ionization electron, e
         * @param nEff effective principal quantum number, unitless
         * @param eFieldNorm_AU norm of the E-Field strength, in sim.atomicUnit.eField()
         *
         * @returns -1. if the result is larger than the maximum representable value of T_ReturnType, unit: 1/unit_time
         * @attention eFieldNorm_AU > 0, otherwise rate formula not defined
         */
        template<typename T_ReturnType>
        HDINLINE static float_X rateFormula(float_X const Z, float_X const nEff, float_X const eFieldNorm_AU)
        {
            float_X const nEffCubed = pmacc::math::cPow(nEff, 3u);
            float_X const ZCubed = pmacc::math::cPow(Z, 3u);

            float_64 const dBase
                = float_64(4.0_X * math::exp(1._X) * ZCubed / (eFieldNorm_AU * pmacc::math::cPow(nEff, 4u)));
            float_64 const dFromADK = math::pow(dBase, float_64(nEff));

            constexpr T_ReturnType pi = pmacc::math::Pi<T_ReturnType>::value;

            float_64 subTerm = pmacc::math::cPow(dFromADK, 2u)
                               * math::exp(float_64(-2._X * ZCubed / (3._X * nEffCubed * eFieldNorm_AU)));

            // explicitly deal with overflows
            constexpr auto maxValueT_Return = std::numeric_limits<T_ReturnType>::max();
            if(subTerm > maxValueT_Return)
                subTerm = -1.;

            // 1/sim.atomicUnit.time()
            T_ReturnType rateADK_AU = eFieldNorm_AU
                                      / (static_cast<T_ReturnType>(8.) * pi * static_cast<T_ReturnType>(Z))
                                      * static_cast<T_ReturnType>(subTerm);

            // factor from averaging over one laser cycle with LINEAR polarization
            if constexpr(
                u32(T_ADKLaserPolarization) == u32(atomicPhysics::enums::ADKLaserPolarization::linearPolarization))
                rateADK_AU *= math::sqrt(static_cast<T_ReturnType>(3._X * nEffCubed * eFieldNorm_AU / (pi * ZCubed)));

            /* unit: A * 1/atomicUnit_time = A * 1/atomicUnit_time * unit_time / unit_time
             *   = A * [unit_time/atomicUnit_time] * 1/unit_time
             *   = (A * timeConversion) * 1/unit_time
             *   = B * 1/unit_time */
            constexpr auto timeConversion
                = static_cast<T_ReturnType>(picongpu::sim.unit.time() / picongpu::sim.atomicUnit.time());

            // unit: 1/unit_time
            return rateADK_AU * timeConversion;
        }

    public:
        /** field ionization ADK rate for a given electric field strength
         *
         * @tparam T_ReturnType type and precision of return, usually float_32 or float_64
         * @tparam T_ChargeStateDataBox instantiated type of dataBox
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundFreeTransitionDataBox instantiated type of dataBox
         *
         * @param eFieldNorm E-field vector norm, in sim.units.eField()
         * @param ionizationPotentialDepression, in eV
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @returns -1. if the result is larger than the maximum representable value of T_ReturnType, unit: 1/unit_time
         */
        template<
            typename T_ReturnType,
            typename T_ChargeStateDataBox,
            typename T_AtomicStateDataBox,
            typename T_BoundFreeTransitionDataBox>
        HDINLINE static T_ReturnType rateADKFieldIonization(
            float_X const eFieldNorm,
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            if constexpr(picongpu::atomicPhysics::debug::fixedRateMatrix::USE_FIXED_RATE_INSTEAD_OF_RATE_CALCULATION)
                return 0._X;

            // unit_energy
            float_X const eFieldEnergy = FieldEnergy::getEFieldEnergy(eFieldNorm * eFieldNorm);

            // eV
            float_X const ionizationEnergy = DeltaEnergyTransition::get(
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox,
                ionizationPotentialDepression,
                chargeStateDataBox);

            /// @details need to check for ionizationEnergy > 0 since IPD ionization may be disabled
            /// @details will also trigger if guarantees eFieldStrength == 0
            bool const enoughFieldEnergy = (eFieldEnergy >= picongpu::sim.pic.conv().eV2Joule(ionizationEnergy));
            if((ionizationEnergy <= 0) || !enoughFieldEnergy)
                return static_cast<T_ReturnType>(0.);

            auto const v = RateFormulaVariables<T_AtomicStateDataBox, T_BoundFreeTransitionDataBox>(
                ionizationEnergy,
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox);

            // unit: atomicUnit_eField
            float_X const eFieldNorm_AU = sim.pic.conv().eField2auEField(eFieldNorm);

            return rateFormula<T_ReturnType>(v.Z, v.nEff, eFieldNorm_AU);
        }

        /** get maximum field ionization ADK rate for electric field strengths inside the given boundaries
         *
         * @tparam T_ReturnType type and precision of return, usually float_32 or float_64
         * @tparam T_ChargeStateDataBox instantiated type of dataBox
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundFreeTransitionDataBox instantiated type of dataBox
         *
         * @param maxEFieldNorm maximum E-field vector norm, in sim.units.eField()
         * @param minEFieldNorm minimum E-field vector norm, in sim.units.eField()
         * @param ionizationPotentialDepression, in eV
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @returns -1. if the result is larger than the maximum representable value of T_ReturnType, unit: 1/unit_time
         */
        template<
            typename T_ReturnType,
            typename T_ChargeStateDataBox,
            typename T_AtomicStateDataBox,
            typename T_BoundFreeTransitionDataBox>
        HDINLINE static T_ReturnType maximumRateADKFieldIonization(
            float_X const minEFieldNorm,
            float_X const maxEFieldNorm,
            float_X const ionizationPotentialDepression,
            uint32_t const transitionCollectionIndex,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundFreeTransitionDataBox const boundFreeTransitionDataBox)
        {
            if constexpr(picongpu::atomicPhysics::debug::fixedRateMatrix::USE_FIXED_RATE_INSTEAD_OF_RATE_CALCULATION)
                return 0._X;

            // unit_energy
            float_X const maxEFieldEnergy = FieldEnergy::getEFieldEnergy(maxEFieldNorm * maxEFieldNorm);

            // eV
            float_X const ionizationEnergy = DeltaEnergyTransition::get(
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox,
                ionizationPotentialDepression,
                chargeStateDataBox);

            /// @details need to check for ionizationEnergy > 0 since IPD ionization may be disabled
            /// @details will also trigger if guarantees eFieldStrength == 0
            bool const enoughFieldEnergy = (maxEFieldEnergy >= picongpu::sim.pic.conv().eV2Joule(ionizationEnergy));
            if((ionizationEnergy <= 0) || !enoughFieldEnergy)
                return static_cast<T_ReturnType>(0.);

            auto const v = RateFormulaVariables<T_AtomicStateDataBox, T_BoundFreeTransitionDataBox>(
                ionizationEnergy,
                transitionCollectionIndex,
                atomicStateDataBox,
                boundFreeTransitionDataBox);

            float_X const nEffCubed = pmacc::math::cPow(v.nEff, u32(3u));
            float_X const ZCubed = pmacc::math::cPow(v.Z, u32(3u));

            float_X const minEField_AU = picongpu::sim.pic.conv().eField2auEField(minEFieldNorm);
            float_X const maxEField_AU = picongpu::sim.pic.conv().eField2auEField(maxEFieldNorm);

            // theoretical maximum ADK Rate, in atomicUnit_eField
            float_X const F_max = 4._X * ZCubed / (3._X * nEffCubed * (4._X * v.nEff - 3._X));

            // fieldStrength for maximum Rate, in atomicUnit_eField, see Notebook 2024 P.43-48
            float_X F;
            if(v.nEff <= 0.75_X || F_max > maxEField_AU)
            {
                F = maxEField_AU;
            }
            else
            {
                if(F_max > minEField_AU)
                    F = F_max;
                else
                    F = minEField_AU;
            }

            return rateFormula<T_ReturnType>(v.Z, v.nEff, F);
        }
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
