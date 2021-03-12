/* Copyright 2020-2021 Klaus Steiniger
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
#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/ArbitraryOrderFDTD.hpp"
#include "picongpu/fields/MaxwellSolver/ArbitraryOrderFDTDPML/ArbitraryOrderFDTDPML.def"

#include <cstdint>


namespace pmacc
{
    namespace traits
    {
        template<uint32_t T_neighbors>
        struct StringProperties<::picongpu::fields::maxwellSolver::ArbitraryOrderFDTDPML<T_neighbors>>
        {
            static StringProperty get()
            {
                pmacc::traits::StringProperty propList("name", "other");
                propList["param"] = std::string("Arbitrary order FDTD with PML, order ") + std::to_string(T_neighbors);

                return propList;
            }
        };

    } // namespace traits
} // namespace pmacc
