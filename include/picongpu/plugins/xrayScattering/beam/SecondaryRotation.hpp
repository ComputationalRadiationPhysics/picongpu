/* Copyright 2020-2021 Pawel Ordyna
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

#include <iostream>

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            namespace beam
            {
                /** Defines a coordinate system rotation.
                 *
                 * The whole rotation consists of two rotations --- first by the yaw angle
                 *  and then by a the pitch angle.
                 *
                 * @tparam T_ParamClass Param class defining the angles.
                 */
                template<typename T_ParamClass>
                struct SecondaryRotation : T_ParamClass
                {
                    using Params = T_ParamClass;
                    struct ReversedAngles
                    {
                        static constexpr float_X yawAngle = -1.0_X * Params::yawAngle;
                        static constexpr float_X pitchAngle = -1.0_X * Params::pitchAngle;
                    } reversedAngles;

                    using ReverseOperation = SecondaryRotation<ReversedAngles>;

                private:
                    static constexpr float_X xAngle = Params::yawAngle;
                    static constexpr float_X yAngle = Params::pitchAngle;

                    //! X axis rotation (yaw angle).
                    static HDINLINE void xRotation(float3_X& vec)
                    {
                        /* A coordinate change for a vector is equal to the inverse
                         * of its basis transform. When the beam is rotated its coordinate
                         * system rotates as well. So the coordinate transfer to such
                         * a rotated basis is just a rotation by the opposite angle.
                         */
                        float_X cos;
                        float_X sin;
                        pmacc::math::sincos(-1.0_X * xAngle, sin, cos);
                        float_X y = vec[1] * cos - vec[2] * sin;
                        float_X z = vec[1] * sin + vec[2] * cos;
                        vec[1] = y;
                        vec[2] = z;
                    }


                    //! Y axis rotation (pitch angle).
                    static HDINLINE void yRotation(float3_X& vec)
                    {
                        float_X cos;
                        float_X sin;
                        pmacc::math::sincos(-1.0_X * yAngle, sin, cos);
                        float_X x = vec[0] * cos + vec[2] * sin;
                        float_X z = -1.0_X * vec[0] * sin + vec[2] * cos;
                        vec[0] = x;
                        vec[2] = z;
                    }

                public:
                    //! Coordinate transform into the rotated coordinate system.
                    static HDINLINE float3_X rotate(float3_X const& vec)
                    {
                        float3_X result = vec;
                        yRotation(result);
                        xRotation(result);
                        return result;
                    }
                };

            } // namespace beam
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
