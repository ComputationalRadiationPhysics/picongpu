/* Copyright 2020-2023 Klaus Steiniger, Sergei Bastrakov
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

#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/ArbitraryOrderFDTD.def"
#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/Derivative.hpp"
#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/Weights.hpp"
#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/MaxwellSolver/DispersionRelation.hpp"
#include "picongpu/fields/MaxwellSolver/GetTimeStep.hpp"
#include "picongpu/fields/differentiation/Curl.hpp"

#include <pmacc/traits/GetStringProperties.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Specialization of the CFL condition checker for the arbitrary-order FDTD
             *
             * @tparam T_neighbors number of neighbors used to calculate the derivatives
             */
            template<uint32_t T_neighbors>
            struct CFLChecker<ArbitraryOrderFDTD<T_neighbors>>
            {
                /** Check the CFL condition, throws when failed
                 *
                 * * @return upper bound on `c * dt` due to chosen cell size according to CFL condition
                 */
                float_X operator()() const
                {
                    // The equations are given at https://picongpu.readthedocs.io/en/latest/models/AOFDTD.html
                    auto weights = aoFDTD::AOFDTDWeights<T_neighbors>{};
                    auto additionalFactor = 0.0_X;
                    for(uint32_t i = 0u; i < T_neighbors; i++)
                    {
                        auto const term = (i % 2) ? -weights[i] : weights[i];
                        additionalFactor += term;
                    }
                    constexpr auto cellSizeSimDim = sim.pic.getCellSize().shrink<simDim>();
                    constexpr float_X invCellSizeSquaredSum
                        = (1.0 / (cellSizeSimDim * cellSizeSimDim)).sumOfComponents();
                    auto const invCorrectedCell2Sum = invCellSizeSquaredSum * additionalFactor * additionalFactor;
                    auto const maxC_DT = 1.0_X / math::sqrt(invCorrectedCell2Sum);
                    constexpr auto dt = getTimeStep();
                    if(sim.pic.getSpeedOfLight() * sim.pic.getSpeedOfLight() * dt * dt * invCorrectedCell2Sum > 1.0_X)
                    {
                        throw std::runtime_error(
                            std::string(
                                "Courant-Friedrichs-Lewy condition check failed, check your simulation.param file\n")
                            + "Courant Friedrichs Lewy condition: c * dt <= " + std::to_string(maxC_DT)
                            + " ? (c * dt = " + std::to_string(sim.pic.getSpeedOfLight() * dt) + ")");
                    }

                    return maxC_DT;
                }
            };

            //! Specialization of the dispersion relation for the arbitrary-order FDTD
            template<uint32_t T_neighbors>
            class DispersionRelation<ArbitraryOrderFDTD<T_neighbors>> : public DispersionRelationBase
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
                    // Dispersion relation is given in https://picongpu.readthedocs.io/en/latest/models/AOFDTD.html
                    auto weights = aoFDTD::AOFDTDWeights<T_neighbors>{};
                    auto rhs = 0.0;
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        auto term = 0.0;
                        for(uint32_t l = 0; l < T_neighbors; l++)
                        {
                            auto const arg = (static_cast<float_64>(l) + 0.5) * absK * direction[d] * step[d];
                            term += weights[l] * math::sin(arg) / step[d];
                        }
                        rhs += term * term;
                    }
                    auto const lhsTerm = math::sin(0.5 * omega * timeStep) / (sim.pic.getSpeedOfLight() * timeStep);
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
                    auto weights = aoFDTD::AOFDTDWeights<T_neighbors>{};
                    auto result = 0.0;
                    for(uint32_t d = 0; d < simDim; d++)
                    {
                        // Calculate d(term^2(absK))/d(absK), where term is from relation()
                        auto term = 0.0;
                        auto termDerivative = 0.0;
                        for(uint32_t l = 0; l < T_neighbors; l++)
                        {
                            auto const arg = (static_cast<float_64>(l) + 0.5) * absK * direction[d] * step[d];
                            term += weights[l] * math::sin(arg) / step[d];
                            termDerivative
                                += weights[l] * (static_cast<float_64>(l) + 0.5) * direction[d] * math::cos(arg);
                        }
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
        template<uint32_t T_neighbors>
        struct StringProperties<::picongpu::fields::maxwellSolver::ArbitraryOrderFDTD<T_neighbors>>
        {
            static StringProperty get()
            {
                pmacc::traits::StringProperty propList("name", "other");
                propList["param"] = std::string("Arbitrary order FDTD, order ") + std::to_string(T_neighbors);

                return propList;
            }
        };

    } // namespace traits
} // namespace pmacc
