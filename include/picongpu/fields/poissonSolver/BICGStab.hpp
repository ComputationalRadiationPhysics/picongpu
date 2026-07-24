/* Copyright 2025 Tapish Narwal, Luca Pennati, Rene Widera
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldTmpOperations.hpp"

namespace picongpu::fields::poissonSolver
{
    struct BICGStab
    {
        // return residual
        // return number of iterations
        void operator()(FieldTmp& fieldV, FieldTmp& fiedlRho, MappingDesc* cellDescription)
        {
            // set boundary conditions on fieldV (Dirichlet or Neuman)

            // normalize the problem based on norm(fieldRho)
        }
    };
} // namespace picongpu::fields::poissonSolver
