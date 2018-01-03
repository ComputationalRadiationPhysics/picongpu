/* Copyright 2015-2018 Axel Huebl, Benjamin Worpitz
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
#include <pmacc/types.hpp>

#include "picongpu/fields/currentInterpolation/None/None.def"

namespace picongpu
{
namespace currentInterpolation
{
using namespace pmacc;

template<uint32_t T_dim>
struct Binomial
{
    static constexpr uint32_t dim = T_dim;

    typedef typename pmacc::math::CT::make_Int<dim, 1>::type LowerMargin;
    typedef typename pmacc::math::CT::make_Int<dim, 1>::type UpperMargin;

    template<typename DataBoxE, typename DataBoxB, typename DataBoxJ>
    HDINLINE void operator()(DataBoxE fieldE,
                             DataBoxB,
                             DataBoxJ fieldJ )
    {
        const DataSpace<dim> self;
        using TypeJ = typename DataBoxJ::ValueType;

        /* 1 2 1 weighting for "left"(1x) "center"(2x) "right"(1x),
         * see Pascal's triangle level N=2 */
        TypeJ dirSum( TypeJ::create(0.0) );
        for( uint32_t d = 0; d < dim; ++d )
        {
            DataSpace<dim> dw;
            dw[d] = -1;
            DataSpace<dim> up;
            up[d] =  1;
            const TypeJ dirDw = fieldJ(dw) + fieldJ(self);
            const TypeJ dirUp = fieldJ(up) + fieldJ(self);

            /* each fieldJ component is added individually */
            dirSum += dirDw + dirUp;
        }

        /* component-wise division by sum of all weightings,
         * in the second order binomial filter these are 4 values per direction
         * (1D: 4 values; 2D: 8 values; 3D: 12 values) */
        const TypeJ filteredJ = dirSum / TypeJ::create(4.0 * dim);

        const float_X deltaT = DELTA_T;
        fieldE(self) -= filteredJ * (float_X(1.0) / EPS0) * deltaT;
    }

    static pmacc::traits::StringProperty getStringProperties()
    {
        pmacc::traits::StringProperty propList( "name", "Binomial" );
        propList["param"] = "period=1;numPasses=1;compensator=false";
        return propList;
    }
};

} /* namespace currentInterpolation */

} /* namespace picongpu */
