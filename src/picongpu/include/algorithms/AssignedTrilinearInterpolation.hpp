/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera
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

#include "pmacc_types.hpp"
#include <boost/type_traits/remove_reference.hpp>
#include <result_of_Functor.hpp>

// forward declaration
namespace picongpu
{
struct AssignedTrilinearInterpolation;
}

namespace PMacc
{
namespace result_of
{

template<typename Cursor>
struct Functor<picongpu::AssignedTrilinearInterpolation, Cursor>
{
    typedef typename
    boost::remove_reference<typename Cursor::type>::type type;
};

} // result_of
} // PMacc

namespace picongpu
{


struct AssignedTrilinearInterpolation
{

    /** Does a 3D trilinear field-to-point interpolation for
     * arbitrary assignment function and arbitrary field_value types.
     *
     * \tparam AssignmentFunction function for assignment
     * \tparam Begin lower margin for interpolation
     * \param End upper margin for interpolation
     *
     * \param cursor cursor pointing to the field
     * \param pos position of the interpolation point
     * \return sum over: field_value * assignment
     *
     * interpolate on grid points in range [Begin;End]
     */
    template<class AssignmentFunction,int Begin,int End,class Cursor >
    static HDINLINE
    typename ::PMacc::result_of::Functor<AssignedTrilinearInterpolation, Cursor>::type
    interpolate(const Cursor& cursor, const float3_X & pos)
    {
        typedef typename ::PMacc::result_of::Functor<AssignedTrilinearInterpolation, Cursor>::type type;

        type result_z = type(0.0);
#pragma unroll 4
        for (float_X z = Begin; z <= End; z += float_X(1.0))
        {
            type result_y = type(0.0);
#pragma unroll 4
            for (float_X y = Begin; y <= End; y += float_X(1.0))
            {
                type result_x = type(0.0);
#pragma unroll 4
                for (float_X x = Begin; x <= End; x += float_X(1.0))
                    //a form factor is the "amount of particle" that is affected by this cell
                    //so we have to sum over: cell_value * form_factor
                    result_x += *cursor(x, y, z) * AssignmentFunction()(x - pos.x());

                result_y += result_x * AssignmentFunction()(y - pos.y());
            }

            result_z += result_y * AssignmentFunction()(z - pos.z());
        }
        return result_z;
    }

    /** Implementation for 2D position*/
    template<class AssignmentFunction, int Begin, int End, class Cursor >
    static HDINLINE
    typename ::PMacc::result_of::Functor<AssignedTrilinearInterpolation, Cursor>::type
    interpolate(const Cursor& cursor, const float2_X & pos)
    {
        typedef typename ::PMacc::result_of::Functor<AssignedTrilinearInterpolation, Cursor>::type type;


        type result_y = type(0.0);
#pragma unroll 4
        for (float_X y = Begin; y <= End; y += float_X(1.0))
        {
            type result_x = type(0.0);
#pragma unroll 4
            for (float_X x = Begin; x <= End; x += float_X(1.0))
                //a form factor is the "amount of particle" that is affected by this cell
                //so we have to sum over: cell_value * form_factor
                result_x += *cursor(x, y ) * AssignmentFunction()(x - pos.x());

            result_y += result_x * AssignmentFunction()(y - pos.y());
        }
        return result_y;
    }

    static PMacc::traits::StringProperty getStringProperties()
    {
        PMacc::traits::StringProperty propList( "name", "uniform" );
        return propList;
    }
};

} // picongpu
