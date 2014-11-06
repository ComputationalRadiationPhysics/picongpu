/**
 * Copyright 2013 Heiko Burau, Rene Widera, Richard Pausch
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


#include <stdint.h>


#ifndef PARTICLE_RPAUSCH
#define PARTICLE_RPAUSCH



#include "math/Vector.hpp"
#include "utilities.hpp"
#include "taylor.hpp"
#include "assert.hpp"

class When
{
    // a enum to describe all needed times
public:

    enum
    {
        first = 0u, now = 1u, old = 2u, older = 3u
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
        location_begin = When::now, momentum_begin = When::now, beta_begin = When::first
    };
    const vec1 momentum_now;
    const vec1 momentum_old;
    const vec1 location_now;
    const picongpu::float_X mass;

public:
    //////////////////////////////////////////////////////////////////
    // constructors:

  HDINLINE Particle(const vec1& locationNow_set, const vec1& momentumOld_set, const vec1& momentumNow_set, const picongpu::float_X mass_set)
    : location_now(locationNow_set), momentum_old(momentumOld_set), momentum_now(momentumNow_set), mass(mass_set)
    {

    }


    //////////////////////////////////////////////////////////////////
    // getters:

    template<unsigned int when>
    HDINLINE vec2 get_location(void) const;
    // get location at time when

    template<unsigned int when>
    HDINLINE vec2 get_momentum(void) const;
    // get momentum at time when

    template<unsigned int when>
    HDINLINE vec2 get_beta(void) const
    {
        return calc_beta(get_momentum<when > ());
    } // get beta at time when except:
    // first --> is specialized below

    template<unsigned int when>
    HDINLINE numtype2 get_gamma(void) const
    {
        return calc_gamma(get_momentum<when > ());
    } // get gamma at time when

    template<unsigned int when>
    HDINLINE numtype2 get_gamma_inv_square(void) const
    {
        return calc_gamma_inv_square(get_momentum<when > ());
    } // get 1/gamma^2

    template< unsigned int when>
    HDINLINE numtype2 get_cos_theta(const vec2& n) const
    {
        // get cos(theta) at time when
        const vec2 beta = get_beta<when > ();
        return calc_cos_theta(n, beta);
    }


private:
    //////////////////////////////////////////////////////////////////
    // private methods:

    HDINLINE vec2 calc_beta(const vec1& momentum) const
    {
        // returns beta=v/c
        const numtype1 gamma1 = calc_gamma(momentum);
        return momentum * (1.0 / (mass * picongpu::SPEED_OF_LIGHT * gamma1));
    }

    HDINLINE numtype2 calc_gamma(const vec1& momentum) const
    {
        // return gamma = E/(mc^2)
        const numtype1 x = util::square<vec1, numtype1 > (momentum * (1.0 / (mass * picongpu::SPEED_OF_LIGHT)));
        return picongpu::math::sqrt(1.0 + x);

    }

    HDINLINE numtype2 calc_gamma_inv_square(const vec1& momentum) const
    {
        // returns 1/gamma^2 = m^2*c^2/(m^2*c^2 + p^2)
        const numtype1 Emass = mass * picongpu::SPEED_OF_LIGHT;
        return Emass / (Emass + (util::square<vec1, numtype1 > (momentum)) / Emass);
    }

    HDINLINE numtype2 calc_cos_theta(const vec2& n, const vec2& beta) const
    {
        // return cos of angle between looking and flight direction
        return (n * beta) / (std::sqrt(beta * beta));
    }


    // setters:

    HDINLINE numtype2 summand(void) const
    {
        // return \vec n independend summand (next value to add to \vec n independend sum)
        const numtype2 x = get_gamma_inv_square<When::now > ();
        return Taylor()(x);
    }

}; // end of Particle definition



template<>
HDINLINE vec2 Particle::get_location<When::now>(void) const
{
    return location_now;
} // get location at time when

template<>
HDINLINE vec2 Particle::get_momentum<When::now>(void) const
{
    return momentum_now;
} // get momentum at time when

template<>
HDINLINE vec2 Particle::get_momentum<When::old>(void) const
{
    return momentum_old;
} // get momentum at time when


#endif
