/* Copyright 2020-2021 Klaus Steiniger, Sergei Bastrakov
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

#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/ArbitraryOrderFDTD.def"
#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/Derivative.hpp"
#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/Weights.hpp"
#include "picongpu/fields/MaxwellSolver/CFLChecker.hpp"
#include "picongpu/fields/differentiation/Curl.hpp"

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
                //! Check the CFL condition, throws when failed
                void operator()() const
                {
                    // The equations are given at https://picongpu.readthedocs.io/en/latest/models/AOFDTD.html
                    auto weights = aoFDTD::AOFDTDWeights<T_neighbors>{};
                    auto additionalFactor = 0.0_X;
                    for(uint32_t i = 0u; i < T_neighbors; i++)
                    {
                        auto const term = (i % 2) ? -weights[i] : weights[i];
                        additionalFactor += term;
                    }
                    if(SPEED_OF_LIGHT * SPEED_OF_LIGHT * DELTA_T * DELTA_T * INV_CELL2_SUM * additionalFactor
                           * additionalFactor
                       > 1.0_X)
                        throw std::runtime_error(
                            "Courant-Friedrichs-Lewy condition check failed, check your grid.param file");
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
