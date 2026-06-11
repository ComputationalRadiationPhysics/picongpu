/*
 * SPDX-FileCopyrightText: Jakob Trojok
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/particles/traits/GetShape.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** defining traits for current assignment
             *
             * @tparam T_DestSpecies type or name as PMACC_CSTRING of the electron species to be created
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
             * @tparam T_DestSpecies type or name as PMACC_CSTRING of the electron species to be created
             * @tparam T_Dim dimension of simulation
             */
            template<typename T_DestSpecies, unsigned T_Dim>
            struct JIonizationAssignment;

            /** 3d case
             */
            template<typename T_DestSpecies>
            struct JIonizationAssignment<T_DestSpecies, DIM3> : public JIonizationAssignmentParent<T_DestSpecies>
            {
                /** functor for  assigning current to databox
                 *
                 * @tparam T_JBox type of current density data box
                 */
                template<typename T_Worker, typename T_JBox>
                HDINLINE void operator()(
                    T_Worker const& worker,
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
                                    alpaka::atomicAdd(
                                        worker.getAcc(),
                                        &(jBoxPar(DataSpace<DIM3>(x, y, z))[i]),
                                        jGridx[i]);
                                }
                            }
                        }
                    }
                }
            };

            /** 2d case
             */
            template<typename T_DestSpecies>
            struct JIonizationAssignment<T_DestSpecies, DIM2> : public JIonizationAssignmentParent<T_DestSpecies>
            {
                /** functor for assigning current to databox
                 */
                template<typename T_Worker, typename T_JBox>
                HDINLINE void operator()(
                    T_Worker const& worker,
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
                                alpaka::atomicAdd(worker.getAcc(), &(jBoxPar(DataSpace<DIM2>(x, y))[i]), jGridx[i]);
                            }
                        }
                    }
                }
            };

            /**@}*/
        } // namespace ionization
    } // namespace particles
} // namespace picongpu
