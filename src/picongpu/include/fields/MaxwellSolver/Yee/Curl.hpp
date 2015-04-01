/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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



#ifndef YEE_CURL_HPP
#define YEE_CURL_HPP

#include "types.h"

namespace picongpu
{
namespace yeeSolver
{
    using namespace PMacc;

    template<class Difference>
    struct Curl
    {
        typedef typename Difference::OffsetOrigin LowerMargin;
        typedef typename Difference::OffsetEnd UpperMargin;

        template<class Memory >
        HDINLINE typename Memory::ValueType operator()(const Memory & mem) const
        {
            Difference diff;
            return float3_X(diff(mem, 1).z() - diff(mem, 2).y(),
                               diff(mem, 2).x() - diff(mem, 0).z(),
                               diff(mem, 0).y() - diff(mem, 1).x());
        }
    };
} // namespace yeeSolver
} // namespace picongpu

#endif	/* YEE_CURL_HPP */

