/* Copyright 2020-2023 Jakob Trojok
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldE.hpp"
#include "picongpu/particles/ParticlesFunctors.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/IonizerReturn.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/JIonizationAssignment.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/JIonizationCalc.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /**@{*/
            /** Implementation of actual ionization current
             *
             * @tparam T_DestSpecies type or name as PMACC_CSTRING of the electron species to be created
             * @tparam T_Dim dimension of simulation
             */
            template<typename T_DestSpecies, unsigned T_Dim>
            struct IonizationCurrent<T_DestSpecies, T_Dim, current::EnergyConservation>
            {
                using ValueType_E = FieldE::ValueType;

                /** Ionization current routine
                 *
                 * @tparam T_JBox type of current density data box
                 */
                template<typename T_Worker, typename T_JBox>
                HDINLINE void operator()(
                    IonizerReturn retValue,
                    float_X const weighting,
                    T_JBox jBoxPar,
                    ValueType_E eField,
                    T_Worker const& worker,
                    floatD_X const pos)
                {
                    /* If there is no ionization, the ionization energy is zero. In that case, there is no need for an
                     * ionization current. */
                    if(retValue.ionizationEnergy != 0.0_X)
                    {
                        auto ionizationEnergy = weighting * retValue.ionizationEnergy * SI::ATOMIC_UNIT_ENERGY
                            / sim.unit.energy(); // convert to PIConGPU units
                        /* calculate ionization current at particle position */
                        float3_X jIonizationPar = JIonizationCalc{}(ionizationEnergy, eField);
                        /* assign ionization current to grid points */
                        JIonizationAssignment<T_DestSpecies, simDim>{}(worker, jIonizationPar, pos, jBoxPar);
                    }
                }
            };

            /** Ionization current deactivated
             */
            template<typename T_DestSpecies, unsigned T_Dim>
            struct IonizationCurrent<T_DestSpecies, T_Dim, current::None>
            {
                using ValueType_E = FieldE::ValueType;

                /** no ionization current
                 */
                template<typename T_Worker, typename T_JBox>
                HDINLINE void operator()(
                    IonizerReturn,
                    float_X const,
                    T_JBox,
                    ValueType_E,
                    T_Worker const&,
                    floatD_X const)
                {
                }
                /**@}*/
            };
        } // namespace ionization
    } // namespace particles
} // namespace picongpu
