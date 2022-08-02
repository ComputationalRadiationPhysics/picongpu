/* Copyright 2021-2022 Sergei Bastrakov
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
#include "picongpu/fields/MaxwellSolver/DispersionRelation.hpp"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.hpp"
#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"
#include "picongpu/fields/MaxwellSolver/None/None.def"
#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"
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
                        addCurrent();
                        this->updateAfterCurrent(currentStepAndSubstep);
                    }
                }

            private:
                /** Interpolate current and add its contribution to the field
                 *
                 * This function assumes fieldJ already has values after synchronization between ranks.
                 * It is always true given the computational loop structure as we always call it inside
                 * update_afterCurrent() and so (some) currents were already added.
                 */
                void addCurrent()
                {
                    using namespace pmacc;
                    using SpeciesWithCurrentSolver =
                        typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, current<>>::type;
                    constexpr auto numSpeciesWithCurrentSolver = bmpl::size<SpeciesWithCurrentSolver>::type::value;
                    constexpr auto existsCurrent = numSpeciesWithCurrentSolver > 0;
                    if constexpr(existsCurrent)
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName(), true);
                        constexpr auto area = type::CORE + type::BORDER;
                        auto const kind = currentInterpolation::CurrentInterpolation::get().kind;
                        if(kind == currentInterpolation::CurrentInterpolation::Kind::None)
                            fieldJ.addCurrentToEMF<area>(currentInterpolation::None{});
                        else
                            fieldJ.addCurrentToEMF<area>(currentInterpolation::Binomial{});
                    }
                }
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