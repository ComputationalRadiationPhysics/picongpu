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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file implements method of processClass specific initialization of newly spawned
 *   ionization macro electrons
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/DeltaEnergyTransition.hpp"
#include "picongpu/particles/atomicPhysics2/initElectrons/CoMoving.hpp"
#include "picongpu/particles/atomicPhysics2/initElectrons/Inelastic2BodyCollisionFromCoMoving.hpp"
#include "picongpu/particles/atomicPhysics2/enums/ProcessClass.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::initElectrons
{
    //! initialization of spawned ionization electrons, generic interface
    template<picongpu::particles::atomicPhysics2::enums::ProcessClass processClass>
    struct InitIonizationElectron
    {
        /** call operator
         *
         * @param ion ion Particle pointing to valid ion frame slot, to init electron from
         * @param electron electron Particle to init, must point to valid electron frame slot
         * @param cascadeIndex index of electron in cascade if multiple electrons are
         *  created by one process
         */
        template<typename T_IonParticle, typename T_ElectronParticle>
        HDINLINE void operator()(
            T_IonParticle& ion,
            T_ElectronParticle& electron,
            uint8_t const cascadeIndex
            //, ...
        ) const;
    };

    /** specialisation for electronicIonization
     *
     * @attention not momentum conserving! We do not model as three body inelastic
     *  collisions, since interacting free electron momentum vector is unknown without
     *  binary pairing
     *
     * @todo implement three body inelastic collision, Brian Marre, 2023
     */
    template<>
    struct InitIonizationElectron<
        picongpu::particles::atomicPhysics2::enums ::ProcessClass::electronicIonization>
    {
        //! call operator
        template<typename T_IonParticle, typename T_ElectronParticle>
        HDINLINE void operator()(T_IonParticle& ion, T_ElectronParticle& electron) const
        {
            /// @todo sample three body inelastic collision for ionization for ionization, Brian Marre, 2023
            CoMoving::init<T_IonParticle, T_ElectronParticle>(ion, electron);
        }
    };

    //! specialisation for autonomousIonization
    template<>
    struct InitIonizationElectron<
        picongpu::particles::atomicPhysics2::enums ::ProcessClass::autonomousIonization>
    {
        //! call operator
        template<
            typename T_Worker,
            typename T_IonParticle,
            typename T_ElectronParticle,
            typename T_AtomicStateDataBox,
            typename T_AutonomousTransitionDataBox,
            typename T_SuperCellLocalOffset,
            typename T_RngGeneratorFactoryFloat,
            typename T_ChargeStateDataBox>
        HDINLINE void operator()(
            T_Worker const& worker,
            T_IonParticle& ion,
            T_ElectronParticle& electron,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_AutonomousTransitionDataBox const autonomousTransitionDataBox,
            T_SuperCellLocalOffset const superCellLocalOffset,
            T_RngGeneratorFactoryFloat& rngFactory,
            T_ChargeStateDataBox const chargeStateDataBox) const
        {
            uint32_t const transitionIndex = ion[transitionIndex_];

            using CollectionIdx = typename T_AutonomousTransitionDataBox::S_TransitionDataBox::Idx;
            using ConfigNumberIdx = typename T_AtomicStateDataBox::Idx;
            using ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

            auto rngGenerator = rngFactory(worker, superCellLocalOffset);

            float_X const deltaEnergy = picongpu::particles::atomicPhysics2::DeltaEnergyTransition ::
                get<T_AtomicStateDataBox, T_AutonomousTransitionDataBox, T_ChargeStateDataBox>(
                    transitionIndex,
                    atomicStateDataBox,
                    autonomousTransitionDataBox,
                    chargeStateDataBox);

            Inelastic2BodyCollisionFromCoMoving::init<T_IonParticle, T_ElectronParticle>(
                ion,
                electron,
                deltaEnergy,
                rngGenerator);
        }
    };
} // namespace picongpu::particles::atomicPhysics2::initElectrons
