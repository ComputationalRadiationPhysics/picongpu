/* Copyright 2023 Brian Marre
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

//! @file implements function for getting deltaEnergy of transition

#pragma once

// need atomicPhysics_Debug.param
#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"

#include <pmacc/static_assert.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics
{
    namespace s_enums = picongpu::particles::atomicPhysics::enums;

    struct DeltaEnergyTransition
    {
    private:
        /** actual implementation of ionization energy calculation
         *
         * @attention arguments must fullfill lowerChargeState <= upperChargeState, otherwise wrong result
         */
        template<typename T_ChargeStateDataBox>
        HDINLINE static float_X ionizationEnergyHelper(
            uint8_t const lowerChargeState,
            uint8_t const upperChargeState,
            T_ChargeStateDataBox const chargeStateDataBox)
        {
            if constexpr(picongpu::atomicPhysics::debug::deltaEnergyTransition::IONIZATION_ENERGY_INVERSION_CHECK)
            {
                if(lowerChargeState > upperChargeState)
                {
                    printf("atomicPhysics ERROR: lowerChargeState > upperChargeState in ionizationEnergy call()");
                    return 0._X;
                }
            }

            // eV
            float_X sumIonizationEnergies = 0._X;
            for(uint8_t k = lowerChargeState; k < upperChargeState; k++)
            {
                // eV
                sumIonizationEnergies += static_cast<float_X>(chargeStateDataBox.ionizationEnergy(k));
            }

            // eV
            return sumIonizationEnergies;
        }

    public:
        /** ionizationEnergy from lowerState- to upperState- chargeState
         *
         * @return unit: eV
         */
        template<
            s_enums::ProcessClassGroup T_ProcessClassGroup,
            typename,
            typename T_ChargeStateDataBox,
            typename... T_AddStuff>
        HDINLINE static float_X ionizationEnergy(
            uint8_t const lowerStateChargeState,
            uint8_t const upperStateChargeState,
            float_X const ionizationPotentialDepression,
            T_ChargeStateDataBox const chargeStateDataBox,
            T_AddStuff const...)
        {
            constexpr bool isAutonomousBased
                = (u8(T_ProcessClassGroup) == u8(s_enums::ProcessClassGroup::autonomousBased));
            constexpr bool isBoundFreeBased
                = (u8(T_ProcessClassGroup) == u8(s_enums::ProcessClassGroup::boundFreeBased));

            [[maybe_unused]] float_X ionizationEnergy = 0._X;

            // defintion of lower and upper state depend on transition type
            if constexpr(isBoundFreeBased)
                ionizationEnergy = ionizationEnergyHelper<T_ChargeStateDataBox>(
                    lowerStateChargeState,
                    upperStateChargeState,
                    chargeStateDataBox);
            else
            {
                // for autonomousBased transitions the charge(upperState) < charge(lowerState), therefore the summation
                //  over ionization energies must be done in reverse and the switch in direction accounted for by
                //  negation
                if constexpr(isAutonomousBased)
                {
                    ionizationEnergy = -ionizationEnergyHelper<T_ChargeStateDataBox>(
                        upperStateChargeState,
                        lowerStateChargeState,
                        chargeStateDataBox);
                }
                else
                {
                    printf("atomicPhysics ERROR: unknonwn transition type, %u", u32(T_ProcessClassGroup));
                }
            }

            return ionizationEnergy - ionizationPotentialDepression;
        }

        /** get deltaEnergy of transition
         *
         * DeltaEnergy is defined as energy(UpperState) - energy(lowerState) [+ ionizationEnergy],
         *  with lower and upper state as given in charge state box
         *
         * @param atomicStateBox deviceDataBox giving access to atomic state property data
         * @param transitionBox deviceDataBox giving access to transition property data,
         * @param ipdAndChargeStateDataBoxAndAddStuff optional, ionizationPotentialDepression:float_X and deviceDataBox
         * giving access to charge state property data, required if T_ProcessClassGroup is ionizing
         *
         * @return unit: eV
         */
        template<typename T_AtomicStateDataBox, typename T_TransitionDataBox, typename... T_IPDAndChargeStateDataBox>
        HDINLINE static float_X get(
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_TransitionDataBox const transitionDataBox,
            T_IPDAndChargeStateDataBox... ipdAndChargeStateDataBoxAndAddStuff)
        {
            using CollectionIdx = typename T_TransitionDataBox::S_TransitionDataBox::Idx;

            CollectionIdx const lowerStateCollectionIndex
                = transitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);
            CollectionIdx const upperStateCollectionIndex
                = transitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);

            // difference initial and final excitation energy
            // eV
            float_X deltaEnergy = static_cast<float_X>(
                atomicStateDataBox.energy(upperStateCollectionIndex)
                - atomicStateDataBox.energy(lowerStateCollectionIndex));

            constexpr s_enums::ProcessClassGroup processClassGroup = T_TransitionDataBox::processClassGroup;
            constexpr bool isIonizing
                = ((u8(processClassGroup) == u8(s_enums::ProcessClassGroup::boundFreeBased))
                   || (u8(processClassGroup) == u8(s_enums::ProcessClassGroup::autonomousBased)));

            if constexpr(isIonizing)
            {
                using ConfigNumberIdx = typename T_AtomicStateDataBox::Idx;
                using ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

                // ionizing electronic interactive processClassGroup
                ConfigNumberIdx const lowerStateConfigNumber
                    = atomicStateDataBox.configNumber(lowerStateCollectionIndex);
                ConfigNumberIdx const upperStateConfigNumber
                    = atomicStateDataBox.configNumber(upperStateCollectionIndex);

                uint8_t const lowerStateChargeState = ConfigNumber::getChargeState(lowerStateConfigNumber);
                uint8_t const upperStateChargeState = ConfigNumber::getChargeState(upperStateConfigNumber);

                // eV
                deltaEnergy
                    += DeltaEnergyTransition::ionizationEnergy<processClassGroup, T_IPDAndChargeStateDataBox...>(
                        lowerStateChargeState,
                        upperStateChargeState,
                        ipdAndChargeStateDataBoxAndAddStuff...);
            }
            return deltaEnergy;
        }
    };
} // namespace picongpu::particles::atomicPhysics
