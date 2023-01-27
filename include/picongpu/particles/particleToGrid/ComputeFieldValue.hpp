/* Copyright 2021-2022 Pawel Ordyna
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/particles/particleToGrid/CombinedDerive.def"
#include "picongpu/particles/particleToGrid/ComputeGridValuePerFrame.def"

#include <pmacc/Environment.hpp>

#include <optional>


namespace picongpu
{
    namespace particles
    {
        namespace particleToGrid
        {
            /** Compute a derived particle attribute
             *
             * @tparam AREA area to compute values in
             * @tparam T_Solver derived attribute solver
             * @tparam T_Species particle species to derive the attribute from
             * @tparam T_Filter particle filter used to filter contributing particles
             */
            template<uint32_t AREA, typename T_Solver, typename T_Species, typename T_Filter>
            struct ComputeFieldValue
            {
                /** Functor implementation
                 *
                 * @param fieldTmp tmp field for storing the result
                 * @param currentStep current simulation step
                 * @param extraSlotNr fieldTmp memory slot to use for processing additional tmp fields possibly
                 * required for the computation. Can be reused after calling this functor.
                 *
                 * @returns pointer to a TaskEvent for handling a possibly unfinished asynchronous communication task
                 * One should call
                 * @code{c++}
                 *  if (returnedPointer != nullPtr)
                 *      eventSystem::setTransactionEvent(*returnedPointer);
                 *     @endcode
                 * after calling this functor.
                 * Moving this outside of this functor allows for doing it just before the field values are needed and
                 * possibly taking an advantage of parallel execution alongside with other tasks. The nullPtr check is
                 * needed since some specializations of this method may not need it and will return a nullptr instead.
                 */
                HINLINE std::optional<EventTask> operator()(
                    FieldTmp& fieldTmp,
                    uint32_t const& currentStep,
                    uint32_t const& extraSlotNr) const;
            };


            //! Specialization for normal derived attributes
            template<uint32_t AREA, typename T_Species, typename T_Filter, typename... T>
            struct ComputeFieldValue<AREA, ComputeGridValuePerFrame<T...>, T_Species, T_Filter>
            {
                HINLINE std::optional<EventTask> operator()(
                    FieldTmp& fieldTmp,
                    uint32_t const& currentStep,
                    uint32_t const& extraSlotNr) const
                {
                    using Solver = ComputeGridValuePerFrame<T...>;
                    DataConnector& dc = Environment<>::get().DataConnector();
                    /*load particle without copy particle data to host*/
                    auto speciesTmp = dc.get<T_Species>(T_Species::FrameType::getName(), true);

                    fieldTmp.getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType::create(0.0));
                    /*run algorithm*/
                    fieldTmp.template computeValue<AREA, Solver, T_Filter>(*speciesTmp, currentStep);
                    // Particles can contribute to cells in GUARD (due to their shape) this values need to be
                    //  added to the neighbouring GPU BOARDERs.
                    return fieldTmp.asyncCommunication(eventSystem::getTransactionEvent());
                }
            };

            //! Specialization for attributes that are a function of two derived attributes
            template<uint32_t AREA, typename T_Species, typename T_Filter, typename... T>
            struct ComputeFieldValue<AREA, CombinedDeriveSolver<T...>, T_Species, T_Filter>
            {
                HINLINE std::optional<EventTask> operator()(
                    FieldTmp& fieldTmp1,
                    uint32_t const& currentStep,
                    uint32_t const& extraSlotNr) const
                {
                    using CombinedSolver = CombinedDeriveSolver<T...>;

                    DataConnector& dc = Environment<>::get().DataConnector();
                    // Get the second tmp field for the second function argument.
                    auto fieldTmp2 = dc.get<FieldTmp>(FieldTmp::getUniqueId(extraSlotNr), true);
                    // Load particles without copy particle data to host.
                    auto speciesTmp = dc.get<T_Species>(T_Species::FrameType::getName(), true);
                    // initialize both fields with zero
                    fieldTmp1.getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType::create(0.0));
                    fieldTmp2->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType::create(0.0));

                    // Derive the first attribute from the particle data.
                    fieldTmp1.template computeValue<AREA, typename CombinedSolver::BaseAttributeSolver, T_Filter>(
                        *speciesTmp,
                        currentStep);
                    /* Particles can contribute to cells in GUARD (due to their shape) these values need to be
                     * added to the neighbouring GPU BORDERs.
                     * Start adding field contribution from the GUARD to the adjacent GPUs while the second attribute
                     * is computed. */
                    EventTask fieldTmpEvent1 = fieldTmp1.asyncCommunication(eventSystem::getTransactionEvent());
                    // Derive the second attribute from the particle data
                    fieldTmp2->template computeValue<AREA, typename CombinedSolver::ModifierAttributeSolver, T_Filter>(
                        *speciesTmp,
                        currentStep);
                    EventTask fieldTmpEvent2 = fieldTmp2->asyncCommunication(eventSystem::getTransactionEvent());

                    // Wait for the communication between the GPUs to finish.
                    eventSystem::setTransactionEvent(fieldTmpEvent1);
                    eventSystem::setTransactionEvent(fieldTmpEvent2);

                    // Modify the 1st field by the values in the 2nd field according to the ModifyingOperation.
                    fieldTmp1.template modifyByField<AREA, typename CombinedSolver::ModifyingOperation>(*fieldTmp2);
                    return std::nullopt;
                }
            };
        } // namespace particleToGrid
    } // namespace particles
} // namespace picongpu
