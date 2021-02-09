/* Copyright 2014-2021 Alexander Debus, Rene Widera
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
        namespace twtsfast
        {
            /** Auxiliary functions for calculating the TWTS field */
            namespace detail
            {
                template<typename T_Type, typename T_AngleType>
                struct RotateField;

                template<typename T_Type, typename T_AngleType>
                struct RotateField<pmacc::math::Vector<T_Type, 3>, T_AngleType>
                {
                    using result = pmacc::math::Vector<T_Type, 3>;
                    using AngleType = T_AngleType;
                    HDINLINE result operator()(result const& fieldPosVector, AngleType const phi) const
                    {
                        /*  Since, the laser propagation direction encloses an angle of phi with the
                         *  simulation y-axis (i.e. direction of sliding window), the positions vectors are
                         *  rotated around the simulation x-axis before calling the TWTS field functions.
                         *  Note: The TWTS field functions are in non-rotated frame and only use the angle
                         *  phi to determine the required amount of pulse front tilt.
                         *  RotationMatrix[PI/2+phi].(y,z) (180Deg-flip at phi=90Deg since coordinate
                         *  system in paper is oriented the other way round.) */
                        AngleType sinPhi;
                        AngleType cosPhi;
                        pmacc::math::sincos(AngleType(phi), sinPhi, cosPhi);
                        return result(
                            fieldPosVector.x(),
                            -sinPhi * fieldPosVector.y() - cosPhi * fieldPosVector.z(),
                            +cosPhi * fieldPosVector.y() - sinPhi * fieldPosVector.z());
                    }
                };

                template<typename T_Type, typename T_AngleType>
                struct RotateField<pmacc::math::Vector<T_Type, 2>, T_AngleType>
                {
                    using result = pmacc::math::Vector<T_Type, 2>;
                    using AngleType = T_AngleType;
                    HDINLINE result operator()(result const& fieldPosVector, AngleType const phi) const
                    {
                        /*  Since, the laser propagation direction encloses an angle of phi with the
                         *  simulation y-axis (i.e. direction of sliding window), the positions vectors are
                         *  rotated around the simulation x-axis before calling the TWTS field functions.
                         *  Note: The TWTS field functions are in non-rotated frame and only use the angle
                         *  phi to determine the required amount of pulse front tilt.
                         *  RotationMatrix[PI/2+phi].(y,z) (180Deg-flip at phi=90Deg since coordinate
                         *  system in paper is oriented the other way round.) */

                        /*  Rotate 90 degree around y-axis, so that TWTS laser propagates within
                         *  the 2D (x,y)-plane. Corresponding position vector for the Ez-components
                         *  in 2D simulations.
                         *  3D     3D vectors in 2D space (x,y)
                         *  x -->  z
                         *  y -->  y
                         *  z --> -x (Since z=0 for 2D, we use the existing
                         *            TWTS-field-function and set -x=0)
                         *
                         * Explicit implementation in 3D coordinates:
                         * fieldPosVector = float3_64( -fieldPosVector.z( ),       //(Here: ==0)
                         *                              fieldPosVector.y( ),
                         *                              fieldPosVector.x( ) );
                         * fieldPosVector = float3_64( fieldPosVector.x( ),
                         *       -sin(phi)*fieldPosVector.y( )-cos(phi)*fieldPosVector.z(),
                         *       +cos(phi)*fieldPosVector.y( )-sin(phi)*fieldPosVector.z()  );
                         * The 2D implementation here only calculates the last two components.
                         * Note: The x-axis of rotation is fine in 2D, because that component now contains
                         *       the (non-existing) simulation z-coordinate. */
                        AngleType sinPhi;
                        AngleType cosPhi;
                        pmacc::math::sincos(AngleType(phi), sinPhi, cosPhi);
                        return result(
                            -sinPhi * fieldPosVector.y() - cosPhi * fieldPosVector.x(),
                            +cosPhi * fieldPosVector.y() - sinPhi * fieldPosVector.x());
                    }
                };

                template<typename T_Type, typename T_AngleType>
                HDINLINE typename RotateField<T_Type, T_AngleType>::result rotateField(
                    T_Type const& fieldPosVector,
                    T_AngleType const phi)
                {
                    return RotateField<T_Type, T_AngleType>()(fieldPosVector, phi);
                }

            } /* namespace detail */
        } /* namespace twtsfast */
    } /* namespace templates */
} /* namespace picongpu */
