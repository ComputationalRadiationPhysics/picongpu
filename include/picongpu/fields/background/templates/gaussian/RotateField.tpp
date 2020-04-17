/* Copyright 2014-2020 Alexander Debus, Rene Widera
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

#include <pmacc/types.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

namespace picongpu
{
namespace templates
{
namespace gaussian
{
/** Auxiliary functions for calculating the Gaussian laser field */
namespace detail
{

    template <typename T_Type, typename T_AngleType>
    struct RotateField;

    template <typename T_Type, typename T_AngleType>
    struct RotateField<pmacc::math::Vector<T_Type,3>, T_AngleType >
    {
        typedef pmacc::math::Vector<T_Type,3> result;
        typedef T_AngleType AngleType;
        HDINLINE result
        operator()( const result& fieldPosVector,
                    const AngleType phi ) const
        {
            /*  Since, the laser propagation direction encloses an angle of phi with the
             *  simulation y-axis (i.e. direction of sliding window), the positions vectors are
             *  rotated around the simulation x-axis before calling the Gaussian laser field functions.
             *  Note: The Gaussian laser field functions are in the non-rotated frame.
             *  RotationMatrix[-(phi-PI/2)].(y,z)
             */
            return result(
                fieldPosVector.x(),
               +math::sin(AngleType(phi))*fieldPosVector.y()
                    -math::cos(AngleType(phi))*fieldPosVector.z() ,
               +math::cos(AngleType(phi))*fieldPosVector.y()
                    +math::sin(AngleType(phi))*fieldPosVector.z() );
        }

    };

    template <typename T_Type, typename T_AngleType>
    struct RotateField<pmacc::math::Vector<T_Type,2>, T_AngleType >
    {
        typedef pmacc::math::Vector<T_Type,2> result;
        typedef T_AngleType AngleType;
        HDINLINE result
        operator()( const result& fieldPosVector,
                    const AngleType phi ) const
        {
            /*  Since, the laser propagation direction encloses an angle of phi with the
             *  simulation y-axis (i.e. direction of sliding window), the positions vectors are
             *  rotated around the simulation x-axis before calling the Gaussian laser field functions.
             *  Note: The Gaussian field functions are in the non-rotated frame.
             *  RotationMatrix[-(phi-PI/2)].(y,z)
             */

            /*  Corresponding position vector for the position-components
             *  in 2D simulations.
             *  3D     3D vectors in 2D space (x,y)
             *  x -->  x
             *  y -->  y
             *  z -->  z (Since z=0 for 2D, we use the existing
             *            Gaussian-laser-field-function and set z=0)
             *
             * Explicit implementation in 3D coordinates:
             * fieldPosVector = float3_64(  fieldPosVector.x(),
             *                              fieldPosVector.y(),
             *                              fieldPosVector.z() );     //(Here: ==0)
             *
             * fieldPosVector = float3_64( fieldPosVector.x(),
             *       +sin(phi)*fieldPosVector.y()-cos(phi)*fieldPosVector.z(),
             *       +cos(phi)*fieldPosVector.y()+sin(phi)*fieldPosVector.z()  );
             * The 2D implementation here only calculates the first two components and takes z=0.
             */
            return result(
                fieldPosVector.x(),
               +math::sin(AngleType(phi))*fieldPosVector.y() );
        }
    };

    template <typename T_Type, typename T_AngleType>
    HDINLINE typename RotateField<T_Type,T_AngleType>::result
    rotateField( const T_Type& fieldPosVector,
                 const T_AngleType phi )
    {
        return RotateField<T_Type,T_AngleType>()(fieldPosVector,phi);
    }

} /* namespace detail */
} /* namespace gaussian */
} /* namespace templates */
} /* namespace picongpu */
