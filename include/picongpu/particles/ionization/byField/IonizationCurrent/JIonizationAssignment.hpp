/* Copyright 2020-2021 Jakob Trojok
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

#include "picongpu/particles/ParticlesFunctors.hpp"
#include "picongpu/simulation_defines.hpp"
#include "picongpu/fields/FieldJ.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** defining traits for current assignment
             *
             * \tparam T_DestSpecies type or name as boost::mpl::string of the electron species to be created
             */
            template<typename T_DestSpecies>
            struct JIonizationAssignmentParent
            {
                using Shape = typename ::picongpu::traits::GetShape<T_DestSpecies>::type;
                using AssignmentFunction = typename Shape::ChargeAssignmentOnSupport;
                static constexpr int supp = AssignmentFunction::support;
                /*(supp + 1) % 2 is 1 for even supports else 0*/
                static constexpr int begin = -supp / 2 + (supp + 1) % 2;
                static constexpr int end = begin + supp;
            };

            /**@{*/
            /** implementation of current assignment
             *
             * \tparam T_Acc alpaka accelerator type
             * \tparam T_DestSpecies type or name as boost::mpl::string of the electron species to be created
             * \tparam T_Dim dimension of simulation
             */
            template<typename T_Acc, typename T_DestSpecies, unsigned T_Dim>
            struct JIonizationAssignment;

            /** 3d case
             */
            template<typename T_Acc, typename T_DestSpecies>
            struct JIonizationAssignment<T_Acc, T_DestSpecies, DIM3>
                : public JIonizationAssignmentParent<T_DestSpecies>
            {
                /** functor for  assigning current to databox
                 *
                 * \tparam T_JBox type of current density data box
                 */
                template<typename T_JBox>
                HDINLINE void operator()(
                    T_Acc const& acc,
                    float3_X const jIonizationPar,
                    float3_X const pos,
                    T_JBox jBoxPar)
                {
                    /* actual assignment */
                    for(int z = JIonizationAssignmentParent<T_DestSpecies>::begin;
                        z < JIonizationAssignmentParent<T_DestSpecies>::end;
                        ++z)
                    {
                        float3_X jGridz = jIonizationPar;
                        jGridz *= typename JIonizationAssignmentParent<T_DestSpecies>::AssignmentFunction{}(
                            float_X(z) - pos.z());
                        for(int y = JIonizationAssignmentParent<T_DestSpecies>::begin;
                            y < JIonizationAssignmentParent<T_DestSpecies>::end;
                            ++y)
                        {
                            float3_X jGridy = jGridz;
                            jGridy *= typename JIonizationAssignmentParent<T_DestSpecies>::AssignmentFunction{}(
                                float_X(y) - pos.y());
                            for(int x = JIonizationAssignmentParent<T_DestSpecies>::begin;
                                x < JIonizationAssignmentParent<T_DestSpecies>::end;
                                ++x)
                            {
                                float3_X jGridx = jGridy;
                                jGridx *= typename JIonizationAssignmentParent<T_DestSpecies>::AssignmentFunction{}(
                                    float_X(x) - pos.x());
                                for(int i = 0; i <= 2; i++)
                                {
                                    cupla::atomicAdd(acc, &(jBoxPar(DataSpace<DIM3>(x, y, z))[i]), jGridx[i]);
                                }
                            }
                        }
                    }
                }
            };

            /** 2d case
             */
            template<typename T_Acc, typename T_DestSpecies>
            struct JIonizationAssignment<T_Acc, T_DestSpecies, DIM2>
                : public JIonizationAssignmentParent<T_DestSpecies>
            {
                /** functor for assigning current to databox
                 */
                template<typename T_JBox>
                HDINLINE void operator()(
                    T_Acc const& acc,
                    float3_X const jIonizationPar,
                    float2_X const pos,
                    T_JBox jBoxPar)
                {
                    for(int y = JIonizationAssignmentParent<T_DestSpecies>::begin;
                        y < JIonizationAssignmentParent<T_DestSpecies>::end;
                        ++y)
                    {
                        float3_X jGridy = jIonizationPar;
                        jGridy *= typename JIonizationAssignmentParent<T_DestSpecies>::AssignmentFunction{}(
                            float_X(y) - pos.y());
                        for(int x = JIonizationAssignmentParent<T_DestSpecies>::begin;
                            x < JIonizationAssignmentParent<T_DestSpecies>::end;
                            ++x)
                        {
                            float3_X jGridx = jGridy;
                            jGridx *= typename JIonizationAssignmentParent<T_DestSpecies>::AssignmentFunction{}(
                                float_X(x) - pos.x());
                            for(int i = 0; i <= 2; i++)
                            {
                                cupla::atomicAdd(acc, &(jBoxPar(DataSpace<DIM2>(x, y))[i]), jGridx[i]);
                            }
                        }
                    }
                }
            };
            /**@}*/
        } // namespace ionization
    } // namespace particles
} // namespace picongpu
