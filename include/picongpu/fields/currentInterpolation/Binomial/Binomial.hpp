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
#include "picongpu/fields/currentInterpolation/None/None.def"

#include <pmacc/dimensions/DataSpace.hpp>


namespace picongpu
{
namespace currentInterpolation
{

    struct Binomial
    {
        static constexpr uint32_t dim = simDim;

        using LowerMargin = typename pmacc::math::CT::make_Int<
            dim,
            1
        >::type ;
        using UpperMargin = LowerMargin;

        template<
            typename T_DataBoxE,
            typename T_DataBoxB,
            typename T_DataBoxJ
        >
        HDINLINE void operator()(
            T_DataBoxE fieldE,
            T_DataBoxB const,
            T_DataBoxJ const fieldJ
        )
        {
            DataSpace< dim > const self;
            using TypeJ = typename T_DataBoxJ::ValueType;

            /* 1 2 1 weighting for "left"(1x) "center"(2x) "right"(1x),
             * see Pascal's triangle level N=2 */
            TypeJ dirSum( TypeJ::create( 0.0 ) );
            for( uint32_t d = 0; d < dim; ++d )
            {
                DataSpace< dim > dw;
                dw[d] = -1;
                DataSpace< dim > up;
                up[d] =  1;
                TypeJ const dirDw = fieldJ( dw ) + fieldJ( self );
                TypeJ const dirUp = fieldJ( up ) + fieldJ( self );

                /* each fieldJ component is added individually */
                dirSum += dirDw + dirUp;
            }

            /* component-wise division by sum of all weightings,
             * in the second order binomial filter these are 4 values per direction
             * (1D: 4 values; 2D: 8 values; 3D: 12 values)
             */
            TypeJ const filteredJ = dirSum / TypeJ::create( float_X( 4.0 ) * dim );

            constexpr float_X deltaT = DELTA_T;
            fieldE( self ) -= filteredJ * ( float_X( 1.0 ) / EPS0 ) * deltaT;
        }

        static pmacc::traits::StringProperty getStringProperties()
        {
            pmacc::traits::StringProperty propList(
                "name",
                "Binomial"
            );
            propList[ "param" ] = "period=1;numPasses=1;compensator=false";
            return propList;
        }
    };

} // namespace currentInterpolation

namespace traits
{

    /* Get margin of the current interpolation
     *
     * This class defines a LowerMargin and an UpperMargin.
     */
    template< >
    struct GetMargin< picongpu::currentInterpolation::Binomial >
    {
    private:
        using MyInterpolation = picongpu::currentInterpolation::Binomial;

    public:
        using LowerMargin = typename MyInterpolation::LowerMargin;
        using UpperMargin = typename MyInterpolation::UpperMargin;
    };

} // namespace traits
} // namespace picongpu
