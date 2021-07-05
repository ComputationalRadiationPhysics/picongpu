/* Copyright 2021 Pawel Ordyna
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
#include "picongpu/particles/scattering/generic/FreeRng.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace direction
            {
                namespace acc
                {
                    template<typename T_ParamClass>
                    struct Random
                    {
                        static constexpr float_X maxPolar = T_ParamClass::maxPolarAngle;
                        static constexpr float_X minPolar = T_ParamClass::minPolarAngle;
                        static constexpr float_X minAzimuth = 0._X;
                        static constexpr float_X maxAzimuth = 2._X * pmacc::math::Pi<float_X>::value;

                        template<typename T_rng, typename T_Particle>
                        DINLINE void operator()(T_rng& rng, T_Particle& particle, float_X const& density) const
                        {
                            // the probability is the photon path in on step over the free path length(sigma * n)

                            /* Azimuth angle is the angle around the x-axis [0,2PI) and polar angle is the angle around
                             * the z-axis [0-PI) Note that compared to e.g. wikipedia the z and x axis are swapped as
                             * our usual propagation direction is X
                             * but it does not influence anything as the axis can be arbitrarily named */
                            float_X azimuthAngle = rng() * float_X(maxAzimuth - minAzimuth)
                                + float_X(minAzimuth);
                            // To get an even distribution on a unit sphere we need to modify the distribution of the
                            // polar angle using arccos.
                            float_X polarAngle;
                            // TODO: Calculate max angle this is valid for
                            if(std::is_same<float_X, float_32>::value && minPolar == 0.
                               && maxPolar <= 1e-2)
                            {
                                // For float32 we don't get small angles as we'd need to calculate the arccos around 1
                                // where the possible float values are sparse. For small angle we can approximate the
                                // density distribution and arccos around 0, else we need to use double precision
                                polarAngle = math::sqrt<sqrt_X>(rng()) * maxPolar;
                            }
                            else
                            {
                                // Optimization with compile-time evaluated ternary for common case
                                const float_64 minPolarCos = (minPolar == 0.)
                                    ? float_64(1.)
                                    : math::cos(float_64(minPolar));
                                const float_64 maxPolarCos = math::cos(float_64(maxPolar));
                                // Note that cos(minPolar)>=cos(maxPolor) for minPolar<=maxPolar
                                polarAngle
                                    = math::acos<float_64>(rng() * (minPolarCos - maxPolarCos) + maxPolarCos);
                            }
                            /* Now we have the azimuth and polar angles by which we want to change the current
                             * direction. So we need some rotations: Assume old direction = A, new direction = B, |A| =
                             * 1 There is a rotation matrix R_A so that A = R_A * e_X (with e_X = [1,0,0] ) It is also
                             * easy to generate a rotation Matrix R_B which transforms a point in Cartesian coordinates
                             * by the azimuth and polar angle So we can say B = R_A * R_B * R_A^T * A (rotate A back to
                             * e_X, rotate by R_B and rotate again to As coordinate system) This can be simplified as
                             * R_A^T * A = e_X and R_B' = R_B * e_X = [cos(polar), sin(polar)*sin(azimuth),
                             * sin(polar)*cos(azimuth)]
                             * --> B = R_A * R_B'
                             * To get R_A we need to find how to rotate e_X to A which means a rotation with cos(a) =
                             * A*e_X (dot product) around A x e_X (cross product) Formula for the general case is on
                             * wikipedia but can be simplified with cos(a) = A_x, A_y² + A_z² = 1 - A_x² = sin²(a) = (1
                             * + A_x)(1 - A_x) So finally one gets B = {{x,-y,-z},
                             * {y,x+z²/(1+x),-y*z/(1+x)},{+z,-y*z/(1+x),x+y²/(1+x)}}*{cos(t),sin(t)sin(p),sin(t)cos(p)}
                             * With x,y,z=A_x,..., t=polar, p=azimuth. This solved by Wolfram Alpha results in the
                             * following formulas
                             */

                            //TODO: what about 2D?
                            trigo_X sinPolar, cosPolar, sinAzimuth, cosAzimuth;
                            pmacc::math::sincos<trigo_X>(polarAngle, sinPolar, cosPolar);
                            pmacc::math::sincos<trigo_X>(azimuthAngle, sinAzimuth, cosAzimuth);
                            const float3_X p = particle[momentum_];
                            float3_X dir  = p / math::sqrt(pmacc::math::abs2(p));
                            const float_X x = dir.x();
                            const float_X y = dir.y();
                            const float_X z = dir.z();
                            if(math::abs(1 + x) <= std::numeric_limits<float_X>::min())
                            {
                                // Special case: x=-1 --> y=z=0 (unit vector), so avoid division by zero
                                dir.x() = -cosPolar;
                                dir.y() = -sinAzimuth * sinPolar;
                                dir.z() = -cosAzimuth * sinPolar;
                            }
                            else
                            {
                                dir.x() = x * cosPolar - z * cosAzimuth * sinPolar - y * sinAzimuth * sinPolar;
                                dir.y() = y * cosPolar - y * z / (1 + x) * cosAzimuth * sinPolar
                                    + (x + z * z / (1 + x)) * sinAzimuth * sinPolar;
                                dir.z() = z * cosPolar + (x + y * y / (1 + x)) * cosAzimuth * sinPolar
                                    - y * z / (1 + x) * sinAzimuth * sinPolar;
                            }
                            particle[momentum_] = dir * math::sqrt(pmacc::math::abs2(p));
                        }
                    };
                } // namespace acc
            } // namespace direction
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
