/* Copyright 2013-2019 Axel Huebl, Heiko Burau, Rene Widera
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

#include "picongpu/algorithms/DifferenceToUpper.hpp"
#include "picongpu/algorithms/DifferenceToLower.hpp"
#include "picongpu/fields/MaxwellSolver/Yee/Curl.def"

#include <pmacc/types.hpp>


namespace picongpu
{
namespace fields
{
namespace maxwellSolver
{
namespace yee
{

    template< typename  Difference >
    struct Curl
    {
        using LowerMargin = typename Difference::OffsetOrigin;
        using UpperMargin = typename Difference::OffsetEnd;

        template<class Memory >
        HDINLINE typename Memory::ValueType operator()( Memory const & mem ) const
        {
            const typename Difference::template GetDifference< 0 > Dx;
            const typename Difference::template GetDifference< 1 > Dy;
            const typename Difference::template GetDifference< 2 > Dz;

            return float3_X(
                Dy( mem ).z() - Dz( mem ).y(),
                Dz( mem ).x() - Dx( mem ).z(),
                Dx( mem ).y() - Dy( mem ).x()
            );
        }
    };

} // namespace yee
} // namespace maxwellSolver
} // namespace fields
} // namespace picongpu
