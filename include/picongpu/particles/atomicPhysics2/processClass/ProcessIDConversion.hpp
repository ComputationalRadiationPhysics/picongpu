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

#pragma once

#include "picongpu/particles/atomicPhysics2/processClass/ProcessClass.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::processClass
{
    /** constexpr conversion methods for processID to processClass
     *
     * processID is the index of a process in the physicalTransitionSpace for each atomic state
     *
     * Encoding is based on the following indexation, ordering level from highest to lowest,
     * [0. no change is always physical transition index 0, always present]
     *  1. first bound-bound, then bound-free, then autonomous based physical transitions
     *  2. up-ward transitions before down-ward transitions
     *  3. different processes according to their ProcessClass enumeration value order,
     *      lowest to highest, enumerated in the processID, repeating for each 1. and 2.
     *      section
     *
     * processIDs are convertible to processClasses using the conversion methods in
     *  picongpu/particles/atomicPhysics2/ProcessClass.hpp
     *
     * Exp.:
     *  Abbreviations: bb ... bound-bound, bf ... bound-free, auto ... autonomous
     *  Key: (bb|bf|auto)_{(up|down), <transitionCollectionIndex>}[<processID>]
     *
     *  [noChange], bb_{up, 0}[processID 0], bb_{up, 0}[processID 1],
     *              bb_{up, 1}[processID 0], bb_{up, 1}[processID 1],
     *              ...
     *              bb_{down, 0}[processID 0], bb_up{down, 1}[processID 0],
     *              ...
     *              bf_{up, 0}[processID 0], bf_{up, 1}[processID 0],
     *              ...
     *              bf_{down, 0]}[processID 0], bf_{down, 0}[processID 1],
     *              bf_{down, 1]}[processID 0], bf_{down, 1}[processID 1],
     *              ...
     *
     * @attention number physical processes must be kept consistent with NumberPhyiscalTransitions
     * @attention possible processes must be kept consistent with ProcessClass enumeration
     *
     * @tparam electronicExcitation is channel active?
     * @tparam electronicDeexcitation is channel active?
     * @tparam spontaneousDeexcitation is channel active?
     * @tparam autonomousIonization is channel active?
     * @tparam electonicIonization is channel active?
     * @tparam fieldIonization is channel active?
     */
    template<
        bool T_electronicExcitation,
        bool T_electronicDeexcitation,
        bool T_spontaneousDeexcitation,
        bool T_electronicIonization,
        bool T_autonomousIonization,
        bool T_fieldIonization>
    struct ProcessIDConversion
    {
        // noChange transition never shared between different process => no conversion necessary

        /** conversion for downward bound-bound transitions
         *
         * @param processID
         *
         * @returns unsigned integer corresponding to processClass enumeration value
         * @retval <255u> indicates undefined processID for tparams
         */
        HDINLINE static constexpr uint8_t getProcessClassBoundBound_Down(uint32_t const processID)
        {
            if constexpr(T_electronicDeexcitation && T_spontaneousDeexcitation)
            {
                // both electronic- and spontaneous deexcitation are active
                if(processID == static_cast<uint32_t>(0u))
                    return static_cast<uint8_t>(ProcessClass::spontaneousDeexcitation);
                if(processID == static_cast<uint32_t>(1u))
                    return static_cast<uint8_t>(ProcessClass::electronicDeexcitation);
            }
            else
            {
                // only one of T_spontaneousDeexcitation or T_electronicDeexcitation is active
                if(processID == static_cast<uint32_t>(0u))
                {
                    if constexpr(T_spontaneousDeexcitation)
                        return static_cast<uint8_t>(ProcessClass::spontaneousDeexcitation);
                    if constexpr(T_electronicDeexcitation)
                        return static_cast<uint8_t>(ProcessClass::electronicDeexcitation);
                }
            }

            // error: none of them is active or processID unknown
            return static_cast<uint8_t>(255u);
        }

        /** conversion for upward bound-bound transitions
         *
         * @param processID
         *
         * @returns unsigned integer corresponding to processClass enumeration value
         * @retval <255u> indicates undefined processID for tparams
         */
        HDINLINE static constexpr uint8_t getProcessClassBoundBound_Up(uint32_t const processID)
        {
            if constexpr(T_electronicExcitation)
            {
                // both electronic- and spontaneous deexcitation are active
                if(processID == static_cast<uint32_t>(0u))
                    return static_cast<uint8_t>(ProcessClass::electronicExcitation);
            }
            // error: none of them is active or processID unknown
            return static_cast<uint8_t>(255u);
        }

        /** conversion for downward bound-free transitions
         *
         * @param processID
         *
         * @returns unsigned integer corresponding to processClass enumeration value
         * @retval <255u> indicates undefined processID for tparams
         *
         * @todo implement recombination, Brian Marre, 2023
         */
        HDINLINE static constexpr uint8_t getProcessClassBoundFree_Down(uint32_t const processID)
        {
            // error: recombination is not implemented yet
            return static_cast<uint8_t>(255u);
        }

        /** conversion for upward bound-free transitions
         *
         * @param processID
         *
         * @returns unsigned integer corresponding to processClass enumeration value
         * @retval <255u> indicates undefined processID for tparams
         */
        HDINLINE static constexpr uint8_t getProcessClassBoundFree_Up(uint32_t const processID)
        {
            if constexpr(T_electronicIonization && T_fieldIonization)
            {
                // both electronic- and spontaneous deexcitation are active
                if(processID == static_cast<uint32_t>(0u))
                    return static_cast<uint8_t>(ProcessClass::electronicIonization);
                if(processID == static_cast<uint32_t>(1u))
                    return static_cast<uint8_t>(ProcessClass::fieldIonization);
            }
            else
            {
                // only one of T_spontaneousDeexcitation or T_electronicDeexcitation is active
                if(processID == static_cast<uint32_t>(0u))
                {
                    if constexpr(T_electronicIonization)
                        return static_cast<uint8_t>(ProcessClass::electronicIonization);
                    if constexpr(T_fieldIonization)
                        return static_cast<uint8_t>(ProcessClass::fieldIonization);
                }
            }

            // error: none of them is active or processID unknown
            return static_cast<uint8_t>(255u);
        }

        /** conversion for upward autonomous transitions
         *
         * @param processID
         *
         * @returns unsigned integer corresponding to processClass enumeration value
         * @retval <255u> indicates undefined processID for tparams
         */
        HDINLINE static constexpr uint8_t getProcessClassAutonomous_Up(uint32_t const processID)
        {
            // error: autonomous transitions are never upward
            return static_cast<uint8_t>(255u);
        }

        /** conversion for downward autonomous transitions
         *
         * @param processID
         *
         * @returns unsigned integer corresponding to processClass enumeration value
         * @retval <255u> indicates undefined processID for tparams
         */
        HDINLINE static constexpr uint8_t getProcessClassAutonomous_Down(uint32_t const processID)
        {
            if constexpr(T_autonomousIonization)
            {
                if(processID == static_cast<uint32_t>(0u))
                    return static_cast<uint8_t>(ProcessClass::autonomousIonization);
            }
            // error: not active or processID unknown
            return static_cast<uint8_t>(255u);
        }
    };

} // namespace picongpu::particles::atomicPhysics2::processClass
