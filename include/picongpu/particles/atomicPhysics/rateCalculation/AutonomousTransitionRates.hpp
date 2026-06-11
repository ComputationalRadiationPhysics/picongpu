/*
 * SPDX-FileCopyrightText: Brian Marre, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"

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
         * @return unit: 1/sim.unit.time
         */
        template<typename T_AutonomousTransitionDataBox>
        HDINLINE static float_X rateAutonomousIonization(
            uint32_t const transitionCollectionIndex,
            T_AutonomousTransitionDataBox const autonomousTransitionDataBox)
        {
            if constexpr(picongpu::atomicPhysics::debug::fixedRateMatrix::USE_FIXED_RATE_INSTEAD_OF_RATE_CALCULATION)
                return 0._X;

            // 1/sim.unit.time
            return autonomousTransitionDataBox.rate(transitionCollectionIndex);
        }
    };
} // namespace picongpu::particles::atomicPhysics::rateCalculation
