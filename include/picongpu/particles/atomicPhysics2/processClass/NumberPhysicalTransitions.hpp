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

//! @file number of physical transition associated with each transition data entry

#pragma once

namespace picongpu::particles::atomicPhysics2::processClass
{
    /** number of physical transitions associated with each known transition
     *
     * @attention number physical processes must be kept consistent with conversion and enumeration
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
    struct NumberPhysicalTransitions
    {
        //! number of physical transitions per stored down-ward bound-bound transition
        HDINLINE static constexpr uint32_t getFactorBoundBoundDown()
        {
            if constexpr(T_electronicDeexcitation && T_spontaneousDeexcitation)
                return 2u; // both electronic- and spontaneous deexcitation are active
            if constexpr(T_electronicDeexcitation || T_spontaneousDeexcitation)
                return 1u; // only one of them is active
            return 0u; // none of them is active
        }

        //! number of physical transitions per stored up-ward bound-bound transition
        HDINLINE constexpr static uint32_t getFactorBoundBoundUp()
        {
            if constexpr(T_electronicExcitation)
                return 1u;
            return 0u;
        }

        //! number of physical transitions per stored down-ward bound-free transition
        HDINLINE static constexpr uint32_t getFactorBoundFreeDown()
        {
            return 0u; // recombination is not yet implemented
        }

        //! number of physical transitions per stored up-ward bound-free transition
        HDINLINE constexpr static uint32_t getFactorBoundFreeUp()
        {
            if constexpr(T_electronicIonization && T_fieldIonization)
                return 2u; // both electronic and field-ionization are active
            if constexpr(T_electronicIonization || T_fieldIonization)
                return 1u; // only one of them is active
            return 0u; // none of them is active
        }

        //! number of physical transitions per stored down-ward autonomous transition
        HDINLINE static constexpr uint32_t getFactorAutonomous()
        {
            if constexpr(T_autonomousIonization)
                return 1u;
            return 0u;
        }
    };
} // namespace picongpu::particles::atomicPhysics2::processClass
