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
#include "picongpu/fields/differentiation/Curl.hpp"
#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/Derivative.hpp"

#include <cstdint>
#include <string>


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
