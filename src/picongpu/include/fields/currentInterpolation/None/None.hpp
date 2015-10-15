/**
 * Copyright 2015 Axel Huebl, Benjamin Worpitz
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

#include "simulation_defines.hpp"
#include "types.h"

#include "fields/currentInterpolation/None/None.def"

namespace picongpu
{
namespace currentInterpolation
{
using namespace PMacc;

template<uint32_t T_dim>
struct None
{
    BOOST_STATIC_CONSTEXPR uint32_t dim = T_dim;

    typedef typename PMacc::math::CT::make_Int<dim, 0>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<dim, 0>::type UpperMargin;

    template<typename DataBoxE, typename DataBoxB, typename DataBoxJ>
    HDINLINE void operator()(DataBoxE fieldE,
                             DataBoxB,
                             DataBoxJ fieldJ )
    {
        const DataSpace<dim> self;

        const float_X deltaT = DELTA_T;
        fieldE(self) -= fieldJ(self) * (float_X(1.0) / EPS0) * deltaT;
    }

};

} /* namespace currentInterpolation */

} /* namespace picongpu */
