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

/** @file implements internal consistency checks for transition Tuples
 *
 * requirements depend on the transition type:
 *  [bound-bound]: (chargeState(upperState) == chargeState(lowerState))
 *                 and (Energy(upperState) > Energy(lowerState))
 *  [bound-free]:  (chargeState(upperState) < chargeState(lowerState))
 *                 [no assumption of energy ordering of lowerState and upperState]
 *  [autonomous] : chargeState(upperState) < chargeState(lowerState) [consistent with autonomous ionization)]
 *                 and (Energy(upperState) > Energy(lowerState))
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics/atomicData/GetStateFromTransitionTuple.hpp"

#include <cstdint>
#include <exception>
#include <string>

namespace picongpu::particles::atomicPhysics::atomicData
{
    namespace detail
    {
        //! check for charge state larger than Z
        template<uint8_t atomicNumber>
        HINLINE void checkForUnphysicalChargeStates(uint8_t const chargeState, std::string debugOutput)
        {
            if(chargeState > atomicNumber)
                throw std::runtime_error("atomicPhysics ERROR: unphysical " + debugOutput + " charge state");
        }
    } // namespace detail

    /** helper function checking transition tuple for internal consistency
     *
     * @tparam T_TransitionTuple type of transition tuple to check
     * @tparam T_ConfigNumber type of configNumber
     *
     * @attention generic version, only call specialized versions!
     * @attention energy checks must be called separately for autonomous and bound-bound transitions
     *
     * @return passes silently if correct, throws error if invalid transition tuple
     */
    template<typename T_TransitionTuple, typename T_ConfigNumber>
    HINLINE void checkTransitionTuple(T_TransitionTuple transitionTuple)
    {
        throw std::runtime_error("not implemented transition tuple check");
    }

    //! version for bound-bound transitions
    template<typename T_ConfigNumber>
    HINLINE void checkTransitionTuple(BoundBoundTransitionTuple<float_X, uint64_t> transitionTuple)
    {
        uint8_t const lowerChargeState
            = T_ConfigNumber::getChargeState(getLowerStateConfigNumber<uint64_t, float_X>(transitionTuple));
        uint8_t const upperChargeState
            = T_ConfigNumber::getChargeState(getUpperStateConfigNumber<uint64_t, float_X>(transitionTuple));

        detail::checkForUnphysicalChargeStates<T_ConfigNumber::atomicNumber>(lowerChargeState, "lower");
        detail::checkForUnphysicalChargeStates<T_ConfigNumber::atomicNumber>(upperChargeState, "upper");

        if(!(lowerChargeState == upperChargeState))
            throw std::runtime_error("atomicPhysics ERROR: bound-bound transitions lower and upper state must have"
                                     "the same charge state");
    }

    //! version for bound-free transitions
    template<typename T_ConfigNumber>
    HINLINE void checkTransitionTuple(BoundFreeTransitionTuple<float_X, uint64_t> transitionTuple)
    {
        uint8_t const lowerChargeState
            = T_ConfigNumber::getChargeState(getLowerStateConfigNumber<uint64_t, float_X>(transitionTuple));
        uint8_t const upperChargeState
            = T_ConfigNumber::getChargeState(getUpperStateConfigNumber<uint64_t, float_X>(transitionTuple));

        detail::checkForUnphysicalChargeStates<T_ConfigNumber::atomicNumber>(lowerChargeState, "lower");
        detail::checkForUnphysicalChargeStates<T_ConfigNumber::atomicNumber>(upperChargeState, "upper");

        if(!(upperChargeState > lowerChargeState))
            throw std::runtime_error("atomicPhysics ERROR: bound-free transitions lower state chargeState must be > "
                                     "upper state chargeState");
    }

    //! version for autonomous transitions
    template<typename T_ConfigNumber>
    HINLINE void checkTransitionTuple(AutonomousTransitionTuple<uint64_t> transitionTuple)
    {
        uint8_t const lowerChargeState
            = T_ConfigNumber::getChargeState(getLowerStateConfigNumber<uint64_t, float_X>(transitionTuple));
        uint8_t const upperChargeState
            = T_ConfigNumber::getChargeState(getUpperStateConfigNumber<uint64_t, float_X>(transitionTuple));

        detail::checkForUnphysicalChargeStates<T_ConfigNumber::atomicNumber>(lowerChargeState, "lower");
        detail::checkForUnphysicalChargeStates<T_ConfigNumber::atomicNumber>(upperChargeState, "upper");

        if(!(upperChargeState < lowerChargeState))
            throw std::runtime_error("atomicPhysics ERROR: autonomous transitions upper state chargeState must be < "
                                     "lower state chargeState");
    }
} // namespace picongpu::particles::atomicPhysics::atomicData
