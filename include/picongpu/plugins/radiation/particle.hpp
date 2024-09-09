/* Copyright 2013-2023 Heiko Burau, Rene Widera, Richard Pausch
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "taylor.hpp"
#include "utilities.hpp"

#include <pmacc/math/Vector.hpp>

#include <cstdint>


namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            class When
            {
                // a enum to describe all needed times
            public:
                enum
                {
                    first = 0u,
                    now = 1u,
                    old = 2u,
                    older = 3u
                };
            };

            class Particle : protected Taylor // Taylor includes just some methodes (no real derived class)
            {
            public:
                //////////////////////////////////////////////////////////////////
                // data:
                // the first time (in above order) to be stored

                enum
                {
                    location_begin = When::now,
                    momentum_begin = When::now,
                    beta_begin = When::first
                };
                const vector_X momentum_now;
                const vector_X momentum_old;
                const vector_X location_now;
                const picongpu::float_X mass;

            public:
                //////////////////////////////////////////////////////////////////
                // constructors:

                HDINLINE Particle(
                    const vector_X& locationNow_set,
                    const vector_X& momentumOld_set,
                    const vector_X& momentumNow_set,
                    const picongpu::float_X mass_set)
                    : momentum_now(momentumNow_set)
                    , momentum_old(momentumOld_set)
                    , location_now(locationNow_set)
                    , mass(mass_set)
                {
                }


                //////////////////////////////////////////////////////////////////
                // getters:

                template<unsigned int when>
                HDINLINE vector_64 getLocation(void) const;
                // get location at time when

                template<unsigned int when>
                HDINLINE vector_64 getMomentum(void) const;
                // get momentum at time when

                template<unsigned int when>
                HDINLINE vector_64 getBeta(void) const
                {
                    return calcBeta(getMomentum<when>());
                } // get beta at time when except:
                // first --> is specialized below

                template<unsigned int when>
                HDINLINE picongpu::float_64 getGamma(void) const
                {
                    return calcGamma(getMomentum<when>());
                } // get gamma at time when

                template<unsigned int when>
                HDINLINE picongpu::float_64 getGammaInvSquare(void) const
                {
                    return calcGammaInvSquare(getMomentum<when>());
                } // get 1/gamma^2

                template<unsigned int when>
                HDINLINE picongpu::float_64 getCosTheta(const vector_64& n) const
                {
                    // get cos(theta) at time when
                    const vector_64 beta = getBeta<when>();
                    return calcCosTheta(n, beta);
                }


            private:
                //////////////////////////////////////////////////////////////////
                // private methods:

                HDINLINE vector_64 calcBeta(const vector_X& momentum) const
                {
                    // returns beta=v/c
                    const picongpu::float_32 gamma1 = calcGamma(momentum);
                    return momentum * (1.0 / (mass * picongpu::sim.pic.getSpeedOfLight() * gamma1));
                }

                HDINLINE picongpu::float_64 calcGamma(const vector_X& momentum) const
                {
                    // return gamma = E/(mc^2)
                    const picongpu::float_32 x = util::square<vector_X, picongpu::float_32>(
                        momentum * (1.0 / (mass * picongpu::sim.pic.getSpeedOfLight())));
                    return picongpu::math::sqrt(1.0 + x);
                }

                HDINLINE picongpu::float_64 calcGammaInvSquare(const vector_X& momentum) const
                {
                    // returns 1/gamma^2 = m^2*c^2/(m^2*c^2 + p^2)
                    const picongpu::float_32 Emass = mass * picongpu::sim.pic.getSpeedOfLight();
                    return Emass / (Emass + (util::square<vector_X, picongpu::float_32>(momentum)) / Emass);
                }

                HDINLINE picongpu::float_64 calcCosTheta(const vector_64& n, const vector_64& beta) const
                {
                    // return cos of angle between looking and flight direction
                    return (n * beta) / (std::sqrt(beta * beta));
                }


                // setters:

                HDINLINE picongpu::float_64 summand(void) const
                {
                    // return \vec n independend summand (next value to add to \vec n independend sum)
                    const picongpu::float_64 x = getGammaInvSquare<When::now>();
                    return Taylor()(x);
                }

            }; // end of Particle definition


            template<>
            HDINLINE vector_64 Particle::getLocation<When::now>(void) const
            {
                return location_now;
            } // get location at time when

            template<>
            HDINLINE vector_64 Particle::getMomentum<When::now>(void) const
            {
                return momentum_now;
            } // get momentum at time when

            template<>
            HDINLINE vector_64 Particle::getMomentum<When::old>(void) const
            {
                return momentum_old;
            } // get momentum at time when

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
