/* Copyright 2021 Sergei Bastrakov
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

#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.hpp"
#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"
#include "picongpu/fields/MaxwellSolver/None/None.def"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/traits/GetStringProperties.hpp>
#include <pmacc/types.hpp>

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
                        // We still have to check the basic PIC condition c * dt < dx as particles need it
                        CFLChecker<SubsteppingBase>{}();
                        PMACC_CASSERT_MSG(
                            Substepping_field_solver_wrong_number_of_substeps____must_be_at_least_1,
                            numSubsteps >= 1);
                    }

                    //! Type-erased functor type to add current density to E for the given time iteration
                    using CurrentAddFunctor = std::function<void(uint32_t)>;

                    /** Set the functor that adds current density to E for the given time iteration
                     *
                     * This is a hook to be used by the main simulation object.
                     *
                     * @param functor functor instance
                     */
                    void setCurrentAddFunctor(CurrentAddFunctor functor)
                    {
                        currentAddFunctor = functor;
                    }

                protected:
                    //! Functor to add current density to E field
                    CurrentAddFunctor currentAddFunctor;
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
                Substepping(MappingDesc const cellDescription): Base(cellDescription)
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
                Substepping(MappingDesc const cellDescription): Base(cellDescription)
                {
                }

                /** Perform the first part of E and B propagation by a PIC time step.
                 *
                 * Together with update_afterCurrent() forms the full propagation by a PIC time step.
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

                /** Perform the last part of E and B propagation by a PIC time step
                 *
                 * Together with update_beforeCurrent() forms the full propagation by a PIC time step.
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
                        this->currentAddFunctor(currentStep);
                        this->updateAfterCurrent(currentStepAndSubstep);
                    }
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