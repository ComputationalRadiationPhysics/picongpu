/* Copyright 2019-2023 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Sergei Bastrakov, Klaus Steiniger
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
#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"
#include "picongpu/fields/MaxwellSolver/Yee/Yee.def"

#include <pmacc/traits/GetStringProperties.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Specialization of the CFL condition checker for the classic Yee solver
             *
             * @tparam T_Defer technical parameter to defer evaluation
             */
            template<typename T_Defer>
            struct CFLChecker<Yee, T_Defer>
            {
                /** Check the CFL condition, doesn't compile when failed
                 *
                 * @return upper bound on `c * dt` due to chosen cell size according to CFL condition
                 */
                float_X operator()() const
                {
                    // Dependance on T_Defer is required, otherwise this check would have been enforced for each setup
                    constexpr auto dt = getTimeStep();
                    PMACC_CASSERT_MSG(
                        Courant_Friedrichs_Lewy_condition_failure____check_your_grid_param_file,
                        (SPEED_OF_LIGHT * SPEED_OF_LIGHT * dt * dt * INV_CELL2_SUM) <= 1.0 && sizeof(T_Defer*) != 0);

                    return 1.0_X / math::sqrt(INV_CELL2_SUM);
                }
            };

            //! Specialization of the dispersion relation for the classic Yee solver
            template<>
            class DispersionRelation<Yee> : public DispersionRelationBase
            {
            public:
                /** Create a functor with the given parameters
                 *
                 * @param omega angular frequency = 2pi * c / lambda
                 * @param direction normalized propagation direction
                 */
                DispersionRelation(float_64 const omega, float3_64 const direction)
                    : DispersionRelationBase(omega, direction)
                {
                }

                /** Calculate f(absK) in the dispersion relation, see comment to the main template
                 *
                 * @param absK absolute value of the (angular) wave number
                 */
                float_64 relation(float_64 const absK) const
                {
                    // (4.12) in Taflove-Hagness, expressed as rhs - lhs = 0
                    auto rhs = 0.0;
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        auto const arg = 0.5 * absK * direction[d] * step[d];
                        auto const term = math::sin(arg) / step[d];
                        rhs += term * term;
                    }
                    auto const lhsTerm = math::sin(0.5 * omega * timeStep) / (SPEED_OF_LIGHT * timeStep);
                    auto const lhs = lhsTerm * lhsTerm;
                    return rhs - lhs;
                }

                /** Calculate df(absK)/d(absK) in the dispersion relation, see comment to the main template
                 *
                 * @param absK absolute value of the (angular) wave number
                 */
                float_64 relationDerivative(float_64 const absK) const
                {
                    // Term-wise derivative in same order as in relation()
                    auto result = 0.0;
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        // Calculate d(term^2(absK))/d(absK), where term is from relation()
                        auto const arg = 0.5 * absK * direction[d] * step[d];
                        auto const term = math::sin(arg) / step[d];
                        auto const termDerivative = 0.5 * direction[d] * math::cos(arg);
                        result += 2.0 * term * termDerivative;
                    }
                    return result;
                }
            };

        } // namespace maxwellSolver
    } // namespace fields
} // namespace picongpu

namespace pmacc
{
    namespace traits
    {
        template<>
        struct StringProperties<::picongpu::fields::maxwellSolver::Yee>
        {
            static StringProperty get()
            {
                auto propList = ::picongpu::fields::maxwellSolver::Yee::getStringProperties();
                // overwrite the name of the solver (inherit all other properties)
                propList["name"].value = "Yee";
                return propList;
            }
        };

    } // namespace traits
} // namespace pmacc
