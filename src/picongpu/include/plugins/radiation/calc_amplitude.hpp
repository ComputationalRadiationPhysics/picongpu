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
 
#include <iostream>

#pragma once


#include "particle.hpp"


//protected:
// error class for wrong time acces

class Error_Accessing_Time
{
public:

    Error_Accessing_Time(void)
    {
    }
};


struct One_minus_beta_times_n
{
    /// Class to calculate 1-\beta \times \vec n
    /// using the best suiting method depending on energy
    /// to achieve the best numerical results
    /// it will be used as base class for amplitude calculations

    //  Taylor just includes a method, When includes just enum

    HDINLINE numtype1 operator()(const vec2& n, const Particle & particle) const
    {
        // 1/gamma^2:
        
        const numtype2 gamma_inv_square(particle.get_gamma_inv_square<When::now > ());

        //numtype2 value; // storage for 1-\beta \times \vec n

        // if energy is high enough to cause numerical errors ( equals if 1/gamma^2 is closs enought to zero)
        // chose a Taylor approximation to to better calculate 1-\beta \times \vec n (which is close to 1-1)
        // is energy is low, then the Appriximation will acuse a larger error, therfor calculate
        // 1-\beta \times \vec n directly
        if (gamma_inv_square < 0.18) // with 0.18 the relativ error will be below 0.001% for Taylor series of order 5
        {
            
            const numtype2 cos_theta(particle.get_cos_theta<When::now > (n)); // cosinus between looking vector and momentum of particle
            const numtype2 taylor_approx(cos_theta * Taylor()(gamma_inv_square) + (1.0 - cos_theta));
            return  (taylor_approx);
        }
        else
        {
            const vec2 beta(particle.get_beta<When::now > ()); // calc v/c=beta
            return  (1.0 - beta * n);
        }
  
    }
};

struct Retarded_time_1
{
    // interface for combined 'Amplitude_Calc' classes
    // contains more parameters than needed to have the
    // same interface as 'Retarded_time_2'

    HDINLINE numtype2 operator()(const numtype2 t,
                                const vec2& n, const Particle & particle) const
    {
        const vec2 r(particle.get_location<When::now > ()); // location
        return (numtype2) (t - (n * r) / (picongpu::SPEED_OF_LIGHT));
    }

};

template<typename Exponent> // divisor to the power of 'Exponent'
struct Old_Method
{
    /// classical method to calculate the real vector part of the radiation's amplitude
    /// this base class includes both possible interpretations:
    /// with Exponent=Cube the integration over t_ret will be assumed (old FFT)
    /// with Exponent=Square the integration over t_sim will be assumed (old DFT)

    HDINLINE vec2 operator()(const vec2& n, const Particle& particle, const numtype2 delta_t) const
    {
        const vec2 beta(particle.get_beta<When::now > ()); // beta = v/c
        const vec2 beta_dot((beta - particle.get_beta < When::now + 1 > ()) / delta_t); // numeric differentiation (backward difference)
        const Exponent exponent; // instance of the Exponent class // ???is a static class and no instance possible???
         //const One_minus_beta_times_n one_minus_beta_times_n;
        const numtype2 factor(exponent(1.0 / (One_minus_beta_times_n()(n, particle))));
        // factor=1/(1-beta*n)^g   g=2 for DFT and g=3 for FFT
        return (n % ((n - beta) % beta_dot)) * factor;
    }
};

// typedef of all possible forms of Old_Method
//typedef Old_Method<util::Cube<numtype2> > Old_FFT;
typedef Old_Method<util::Square<numtype2> > Old_DFT;




// ------- Calc Amplitude class ------------- //

template<typename TimeCalc, typename VecCalc>
class Calc_Amplitude
{
    /// final class for amplitude calculations
    /// derived from a class to calculate the retarded time (TimeCalc; possibilities:
    /// Retarded_Time_1 and Retarded_Time_2) and from a class to  calculate
    /// the real vector part of the amplitude (VecCalc; possibilities:
    /// Old_FFT, Old_DFT, Partial_Integral_Method_1, Partial_Integral_Method_2)
public:
    /// constructor
    // takes a lot of parameters to have a general interface
    // not all parameters are needed for all possible combinations
    // of base classes

    HDINLINE Calc_Amplitude(const Particle& particle,
                           const numtype2 delta_t,
                           const numtype2 t_sim)
    : particle(particle), delta_t(delta_t), t_sim(t_sim)
    {
    }

    // get real vector part of amplitude

    HDINLINE vec2 get_vector(const vec2& n) const
    {
        const vec2 look_direction(n.unit_vec()); // make sure look_direction is a unit vector
        VecCalc vecC;
        return vecC(look_direction, particle, delta_t);
    }

    // get retarded time

    HDINLINE numtype2 get_t_ret(const vec2 look_direction) const
    {
        TimeCalc timeC;
        return timeC(t_sim, look_direction, particle);

        //  const vec2 r = particle.get_location<When::now > (); // location
        //  return (numtype2) (t - (n * r) / (picongpu::SPEED_OF_LIGHT));
    }

private:
    // data:
    const Particle& particle; // one particle
    const numtype2 delta_t; // length of one timestep in simulation
    const numtype2 t_sim; // simulation time (for methods not using index*delta_t )


};



