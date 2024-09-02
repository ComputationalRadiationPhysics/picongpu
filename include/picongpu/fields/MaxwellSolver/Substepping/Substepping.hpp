/* Copyright 2021-2023 Sergei Bastrakov
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

#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/DispersionRelation.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.hpp"
#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"
#include "picongpu/fields/MaxwellSolver/None/None.def"
#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetStringProperties.hpp>
#include <pmacc/types.hpp>

#include <boost/mpl/size.hpp>

#include <cstdint>
#include <functional>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            namespace substepping
            {
                /** Base class with common implementation details for substepping field solvers
                 *
                 * @tparam T_BaseSolver base field solver, follows requirements of field solvers
                 * @tparam T_numSubsteps number of field solver steps per PIC time iteration
                 */
                template<typename T_BaseSolver, uint32_t T_numSubsteps>
                class SubsteppingBase : public T_BaseSolver
                {
                public:
                    //! Number of substeps
                    static constexpr uint32_t numSubsteps = T_numSubsteps;

                    //! Base field solver type
                    using Base = T_BaseSolver;

                    /** Create base substepping solver instance
                     *
                     * @param cellDescription mapping description for kernels
                     */
                    SubsteppingBase(MappingDesc const cellDescription) : Base(cellDescription)
                    {
                        PMACC_CASSERT_MSG(
                            Substepping_field_solver_wrong_number_of_substeps____must_be_at_least_1,
                            numSubsteps >= 1);
                    }
                };
            } // namespace substepping

            /** Substepping None solver does not make much sense, but is allowed and works same as None.
             *
             * @tparam T_numSubsteps number of substeps per PIC time iteration
             */
            template<uint32_t T_numSubsteps>
            class Substepping<None, T_numSubsteps> : public substepping::SubsteppingBase<None, T_numSubsteps>
            {
            public:
                //! Base type
                using Base = substepping::SubsteppingBase<None, T_numSubsteps>;

                /** Create None substepping solver instance
                 *
                 * @param cellDescription mapping description for kernels
                 */
                Substepping(MappingDesc const cellDescription) : Base(cellDescription)
                {
                }
            };

            /** Substepping FDTD solver
             *
             * @tparam TArgs template parameters for the base FDTD solver
             * @tparam T_numSubsteps number of substeps per PIC time iteration
             */
            template<typename... TArgs, uint32_t T_numSubsteps>
            class Substepping<FDTD<TArgs...>, T_numSubsteps>
                : public substepping::SubsteppingBase<FDTD<TArgs...>, T_numSubsteps>
            {
            public:
                //! Base type
                using Base = substepping::SubsteppingBase<FDTD<TArgs...>, T_numSubsteps>;

                /** Create FDTD substepping solver instance
                 *
                 * @param cellDescription mapping description for kernels
                 */
                Substepping(MappingDesc const cellDescription) : Base(cellDescription)
                {
                    previousJInitialized = false;
                    if(existsCurrent)
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName());
                        auto const& gridBuffer = fieldJ.getGridBuffer();
                        previousJ = pmacc::makeDeepCopy(gridBuffer.getDeviceBuffer());
                    }
                }

                /** Perform the first part of E and B propagation by a PIC time step.
                 *
                 * Does not account for the J term, which will be added by addCurrent().
                 * Together with addCurrent() and update_afterCurrent() forms the full propagation by a PIC time step.
                 *
                 * Here we only do the update_beforeCurrent for the first substep.
                 * However the calling side should not rely on any particular state of fields after this function.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_beforeCurrent(uint32_t const currentStep)
                {
                    this->updateBeforeCurrent(static_cast<float_X>(currentStep));
                }

                /** Add contribution of FieldJ in the given area according to Ampere's law
                 *
                 * The first time addCurrent is called from the main simulation loop, it is default subStep 0.
                 * Later calls happen from inside update_afterCurrent().
                 *
                 * @tparam T_area area to operate on
                 *
                 * @param subStep substep index, in [0, numSubsteps)
                 */
                template<uint32_t T_area>
                void addCurrent(uint32_t const subStep = 0u)
                {
                    if(!existsCurrent)
                        return;

                    // J values in the middle of the current PIC time iteration
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName());

                    /* Initialize previousJ if necessary so that we can process everything uniformly.
                     * The condition can only be true at the very first time step or just after a restart.
                     * For these cases we approximate J with first and not second order of accuracy.
                     * The issue is in principle minor as it only concerns a fixed number of time iterations.
                     *
                     * @TODO Remove this limitation by implementing current deposition at step -1 and saving the
                     * resulting J as previous; save previous J in checkpointing. Note: with existing code base
                     * there can only be a single FieldJ object due to IDs of communications, so checkpointing of
                     * previous J would require some fix or workaround for it.
                     */
                    if(!previousJInitialized)
                        copyToPreviousJ(fieldJ);

                    /* In order to keep the central nature of time derivatives in the field solver (and the consequent
                     * approximation order in time), we have to add contribution of J in the middle of the given
                     * subStep, so at time
                     *     t_sub = currentStep * sim.pic.getDt() + (subStep + 0.5) * sim.pic.getDt() / numSubsteps.
                     * We process each grid point separately and independently, so the following only concerts time.
                     * With Esirkepov/EmZ current deposition we have, assuming sufficient smoothness of J(t):
                     *     fieldJ = J(t_curr) + O(sim.pic.getDt()^2) with t_curr = (currentStep + 0.5) *
                     * sim.pic.getDt(), previousJ = J(t_prev) + O(sim.pic.getDt()^2) with t_prev = (currentStep - 0.5)
                     * * sim.pic.getDt(). Using linear interpolation or extrapolation (depending on subStep) of J(t)
                     * yields J_sub = J(t_prev) + (t_sub - t_prev) / (t_curr - t_prev) * (J(t_curr) - J(t_prev)).
                     * Applying Taylor expansion and again assuming sufficient smoothness of J(t) one can obtain
                     *     J_sub =  J(t_sub) + O(sim.pic.getDt()^2).
                     * Since J(t_curr), J(t_prev) terms appear linearly in J_sub, same approximation order holds when
                     * fieldJ, previousJ are used instead.
                     * Denoting
                     *     alpha = (subStep + 0.5) / numSubsteps,
                     * substepping solver then has to apply the following current density value:
                     *     J = (0.5 - alpha) * previousJ + (0.5 + alpha) * fieldJ.
                     */

                    // Base coefficient in front of J in Ampere's law
                    constexpr float_X baseCoeff = -(1.0_X / EPS0) * getTimeStep();
                    auto const alpha = static_cast<float_X>((subStep + 0.5) / this->numSubsteps);
                    auto const prevCoeff = (0.5_X - alpha) * baseCoeff;
                    auto const currentCoeff = (0.5_X + alpha) * baseCoeff;
                    this->template addCurrentImpl<T_area>(previousJ->getDataBox(), prevCoeff);
                    this->template addCurrentImpl<T_area>(fieldJ.getDeviceDataBox(), currentCoeff);

                    /* After the last substep copy current J to previous.
                     * In case numSubsteps > 1 we are here once per PIC time iteration and T_area == CORE + BORDER.
                     * In case numSubsteps == 1 we may be here more than once, but prevCoeff == 0 and so copy is safe.
                     */
                    if(subStep == this->numSubsteps - 1)
                        copyToPreviousJ(fieldJ);
                }

                /** Perform the last part of E and B propagation by a PIC time step
                 *
                 * Does not account for the J term, which has been added by addCurrent().
                 * Together with addCurrent() and update_beforeCurrent() forms the full propagation by a PIC time step.
                 *
                 * Here we finish the first substep and then iterate over all remaining substeps doing a full update.
                 * However the calling side should not rely on any particular state of fields before this function.
                 * After it is completed, the fields are properly propagated by a PIC time step.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_afterCurrent(uint32_t const currentStep)
                {
                    this->updateAfterCurrent(static_cast<float_X>(currentStep));
                    // By now we made 1 full substep, do the remaining ones
                    for(uint32_t subStep = 1; subStep < this->numSubsteps; subStep++)
                    {
                        auto const currentStepAndSubstep
                            = static_cast<float_X>(currentStep) + static_cast<float_X>(subStep) * getTimeStep();
                        this->updateBeforeCurrent(currentStepAndSubstep);
                        // By now FieldJ has been communicated so we can directly add it
                        addCurrent<type::CORE + type::BORDER>(subStep);
                        this->updateAfterCurrent(currentStepAndSubstep);
                    }
                }

            private:
                /** Copy given current density device values to previousJ
                 *
                 * @param fieldJ current density
                 */
                void copyToPreviousJ(FieldJ& fieldJ)
                {
                    auto& currentGridBuffer = fieldJ.getGridBuffer();
                    previousJ->copyFrom(currentGridBuffer.getDeviceBuffer());
                    previousJInitialized = true;
                }

                //! Buffer type to store previous J values
                using DeviceBufferJ = FieldJ::Buffer::DBuffer;

                /** Device buffer for values of J on previous PIC time step
                 *
                 * Not used when existsCurrent is false.
                 */
                std::unique_ptr<DeviceBufferJ> previousJ;

                //! Typelist of species with current solver
                using SpeciesWithCurrentSolver =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, current<>>::type;
                //! Whether the simulation has any current sources
                static constexpr auto existsCurrent
                    = (pmacc::mp_size<SpeciesWithCurrentSolver>::value > 0) || FieldBackgroundJ::activated;

                /** Whether previousJ is initialized with data of J from previous time step
                 *
                 * The only times it isn't initialized are the very first time step in simulation or after a restart.
                 * In these cases assume previous field J is same as current field J.
                 */
                bool previousJInitialized;
            };

            /** Specialization of the CFL condition checker for substepping solver
             *
             * Uses CFL of the base solver, those already take care of using the correct time (sub)step.
             *
             * @tparam T_BaseSolver base field solver, follows requirements of field solvers
             * @tparam T_numSubsteps number of substeps per PIC time iteration
             * @tparam T_Defer technical parameter to defer evaluation
             */
            template<typename T_BaseSolver, uint32_t T_numSubsteps, typename T_Defer>
            struct CFLChecker<Substepping<T_BaseSolver, T_numSubsteps>, T_Defer> : CFLChecker<T_BaseSolver, T_Defer>
            {
            };

            /** Specialization of the dispersion relation for substepping solver
             *
             * Uses dispersion relation of the base solver, those already take care of using the correct time
             * (sub)step.
             *
             * @tparam T_BaseSolver base field solver, follows requirements of field solvers
             * @tparam T_numSubsteps number of substeps per PIC time iteration
             */
            template<typename T_BaseSolver, uint32_t T_numSubsteps>
            class DispersionRelation<Substepping<T_BaseSolver, T_numSubsteps>>
                : public DispersionRelation<T_BaseSolver>
            {
            public:
                /** Create an instance with the given parameters
                 *
                 * @param omega angular frequency = 2pi * c / lambda
                 * @param direction normalized propagation direction
                 */
                DispersionRelation(float_64 const omega, float3_64 const direction)
                    : DispersionRelation<T_BaseSolver>(omega, direction)
                {
                }
            };

        } // namespace maxwellSolver
    } // namespace fields

    namespace traits
    {
        /** Get margin for any field access in the substepping solver
         *
         * It matches the base solver.
         *
         * @tparam T_BaseSolver base field solver, follows requirements of field solvers
         * @tparam T_numSubsteps number of substeps per PIC time iteration
         * @tparam T_Field field type
         */
        template<typename T_BaseSolver, uint32_t T_numSubsteps, typename T_Field>
        struct GetMargin<picongpu::fields::maxwellSolver::Substepping<T_BaseSolver, T_numSubsteps>, T_Field>
            : GetMargin<T_BaseSolver, T_Field>
        {
        };
    } // namespace traits
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        /** Get string properties for the substepping solver
         *
         * @tparam T_BaseSolver base field solver, follows requirements of field solvers
         * @tparam T_numSubsteps number of substeps per PIC time iteration
         */
        template<typename T_BaseSolver, uint32_t T_numSubsteps>
        struct StringProperties<picongpu::fields::maxwellSolver::Substepping<T_BaseSolver, T_numSubsteps>>
        {
            //! Get string property
            static StringProperty get()
            {
                pmacc::traits::StringProperty propList = StringProperties<T_BaseSolver>::get();
                propList["param"]
                    = std::string("Substepping with ") + std::to_string(T_numSubsteps) + " substeps per PIC time step";
                return propList;
            }
        };
    } // namespace traits
} // namespace pmacc
