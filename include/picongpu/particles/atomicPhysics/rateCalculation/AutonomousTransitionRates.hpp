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


/** @file implements interface methods for autonomous ionization transitions
 *
 * not technically necessary, since current atomicData data base implementation stores
 *  only pre-calculated rates for autonomous transitions.
 * Implemented anyway for consistency of interface with bound-bound/-free and abstraction.
 *
 * spontaneous radiative deexcitation while also a autonomous process is implemented in
 *  BoundBoundTransitionrates.hpp, since it relies on bound-bound transition Data.
 *
 * based on the
 *
 * - I.I.Sobelman, L.A.Vainshtein, E.A.Yukov,
 *  "Excitation of Atoms and Broadening of Spectral Lines", 2nd Ed.
 *  Springer, Berlin, 1995, pp.120-124
 */

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    struct AutonomousTransitionRates
    {
        /** rate of autonomous ionization for a given autonomous transition
         *
         * @tparam T_AutonomousTransitionDataBox instantiated type of dataBox
         *
         * @param transitionCollectionIndex index of transition in autonomousTransitionDataBox
         * @param autonomousTransitionDataBox access to autonomous transition data
         *
         * @return unit: 1/sim.unit.time(), usually Delta_T_SI ... PIC time step length
         */
        template<typename T_AutonomousTransitionDataBox>
        HDINLINE static float_X rateAutonomousIonization(
            uint32_t const transitionCollectionIndex,
            T_AutonomousTransitionDataBox const autonomousTransitionDataBox)
        {
            return autonomousTransitionDataBox.rate(transitionCollectionIndex); // 1/sim.unit.time()
        }
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
