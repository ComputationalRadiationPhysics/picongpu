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
#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"
#include "picongpu/fields/MaxwellSolver/None/None.def"
#include "picongpu/fields/MaxwellSolver/FDTD/FDTD.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/traits/GetStringProperties.hpp>
#include <pmacc/types.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Substepping None solver does not make much sense, but is allowed.
             *
             * @tparam T_numSubSteps number of substeps per PIC time iteration
             */
            template<uint32_t T_numSubSteps>
            class Substepping<None, T_numSubSteps> : None
            {
            public:
                //! Base field solver type
                using Base = None;
            
                Substepping(MappingDesc const cellDescription): Base(cellDescription)
                {
                    // We still have to check the basic PIC c * dt < dx as particles need it
                    CFLChecker<Substepping>{}();
                    PMACC_CASSERT_MSG(
                        Substepping_field_solver_wrong_number_of_substeps____must_be_at_least_1,
                        T_numSubSteps >= 1);
                }
            };

            template<typename T_CurlE, typename T_CurlB, uint32_t T_numSubSteps>
            class Substepping<FDTD<T_CurlE, T_CurlB>, T_numSubSteps> : FDTD<T_CurlE, T_CurlB>
            {
            public:
            
                //! Base field solver type
                using Base = FDTD<T_CurlE, T_CurlB>;
            
                Substepping(MappingDesc const cellDescription): Base(cellDescription)
                {
                    // We still have to check the basic PIC c * dt < dx as particles need it
                    CFLChecker<Substepping>{}();
                    PMACC_CASSERT_MSG(
                        Substepping_field_solver_wrong_number_of_substeps____must_be_at_least_1,
                        T_numSubSteps >= 1);
                }
                
                /** Perform the first part of E and B propagation by a full time step.
                 *
                 * Together with update_afterCurrent( ) forms the full propagation by a time step.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_beforeCurrent(uint32_t const currentStep)
                {
                    Base::update_beforeCurrent(static_cast<float_X>(currentStep), getTimeStep());
                }

                /** Perform the last part of E and B propagation by a time step
                 *
                 * Together with update_beforeCurrent( ) forms the full propagation.
                 *
                 * @param currentStep index of the current time iteration
                 */
                void update_afterCurrent(uint32_t const currentStep)
                {
                    constexpr auto subStepDt = getTimeStep();
                    Base::update_afterCurrent(static_cast<float_X>(currentStep), subStepDt);
                    // By now we made 1 substep, do the rest here
                    for (uint32_t subStep = 1; subStep < T_numSubSteps; subStep++)
                    {
                        auto const currentStepAndSubstep = static_cast<float_X>(currentStep) + subStepDt * static_Cast<float_X>(subStep);
                        Base::update_beforeCurrent(currentStepAndSubstep, subStepDt);
                        // TODO: call currentInterpolationAndAdditionToEMF() here
                        Base::update_afterCurrent(currentStepAndSubstep, subStepDt);
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
         * @tparam T_numSubSteps number of substeps per PIC time iteration
         * @tparam T_Field field type
         */
        template<typename T_BaseSolver, uint32_t T_numSubSteps, typename T_Field>
        struct GetMargin<picongpu::fields::maxwellSolver::Substepping<T_BaseSolver, T_numSubSteps>, T_Field>:
            GetMargin<T_BaseSolver,  T_Field>
        {
        };
    } // namespace traits
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        template<typename T_BaseSolver, uint32_t T_numSubSteps>
        struct StringProperties<picongpu::fields::maxwellSolver::Substepping<T_BaseSolver, T_numSubSteps>>
        {
            static StringProperty get()
            {
                pmacc::traits::StringProperty propList = StringProperties<T_BaseSolver>::get();
                propList["param"] = std::string("Substepping with ") + std::to_string(T_numSubSteps) + " substeps per PIC step";
                return propList;
            }
        };
    } // namespace traits
} // namespace pmacc