/* Copyright 2013-2019 Heiko Burau, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/cuSTL/zone/SphericZone.hpp"
#include <cufft.h>

namespace pmacc
{
namespace algorithm
{
namespace kernel
{

template<>
template<typename Zone, typename DestCursor, typename SrcCursor>
void FFT<2>::operator()(const Zone& p_zone, const DestCursor& destCursor, const SrcCursor& srcCursor)
{
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, p_zone.size.x(), p_zone.size.y(), CUFFT_R2C));

    CUFFT_CHECK(cufftExecR2C(plan, (cufftReal*)&(*(srcCursor(p_zone.offset))),
                        (cufftComplex*)&(*destCursor(p_zone.offset))));

    CUFFT_CHECK(cufftDestroy(plan));
}

} // kernel
} // algorithm
} // pmacc
