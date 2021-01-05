/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch
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

#include "particle.hpp"

#include <iostream>


namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            // protected:
            // error class for wrong time access

            class Error_Accessing_Time
            {
            public:
                Error_Accessing_Time(void)
                {
                }
            };


            struct One_minus_beta_times_n
            {
                /// Class to calculate \f$1-\beta \times \vec n\f$
                /// using the best suiting method depending on energy
                /// to achieve the best numerical results
                /// it will be used as base class for amplitude calculations

                //  Taylor just includes a method, When includes just enum

                HDINLINE picongpu::float_32 operator()(const vector_64& n, const Particle& particle) const
                {
                    // 1/gamma^2:

                    const picongpu::float_64 gamma_inv_square(particle.get_gamma_inv_square<When::now>());

                    // picongpu::float_64 value; // storage for 1-\beta \times \vec n

                    // if energy is high enough to cause numerical errors ( equals if 1/gamma^2 is close enough to
                    // zero) chose a Taylor approximation to to better calculate 1-\beta \times \vec n (which is close
                    // to 1-1) is energy is low, then the approximation will cause a larger error, therefor calculate
                    // 1-\beta \times \vec n directly
                    // with 0.18 the relative error will be below 0.001% for a Taylor series of 1-sqrt(1-x) of 5th
                    // order
                    if(gamma_inv_square < picongpu::GAMMA_INV_SQUARE_RAD_THRESH)
                    {
                        const picongpu::float_64 cos_theta(particle.get_cos_theta<When::now>(
                            n)); // cosine between looking vector and momentum of particle
                        const picongpu::float_64 taylor_approx(
                            cos_theta * Taylor()(gamma_inv_square) + (1.0 - cos_theta));
                        return (taylor_approx);
                    }
                    else
                    {
                        const vector_64 beta(particle.get_beta<When::now>()); // calculate v/c=beta
                        return (1.0 - beta * n);
                    }
                }
            };

            struct Retarded_time_1
            {
                // interface for combined 'Amplitude_Calc' classes
                // contains more parameters than needed to have the
                // same interface as 'Retarded_time_2'

                HDINLINE picongpu::float_64 operator()(
                    const picongpu::float_64 t,
                    const vector_64& n,
                    const Particle& particle) const
                {
                    const vector_64 r(particle.get_location<When::now>()); // location
                    return (picongpu::float_64)(t - (n * r) / (picongpu::SPEED_OF_LIGHT));
                }
            };

            template<typename Exponent> // divisor to the power of 'Exponent'
            struct Old_Method
            {
                /// classical method to calculate the real vector part of the radiation's amplitude
                /// this base class includes both possible interpretations:
                /// with Exponent=Cube the integration over t_ret will be assumed (old FFT)
                /// with Exponent=Square the integration over t_sim will be assumed (old DFT)

                HDINLINE vector_64
                operator()(const vector_64& n, const Particle& particle, const picongpu::float_64 delta_t) const
                {
                    const vector_64 beta(particle.get_beta<When::now>()); // beta = v/c
                    const vector_64 beta_dot(
                        (beta - particle.get_beta<When::now + 1>())
                        / delta_t); // numeric differentiation (backward difference)
                    const Exponent exponent; // instance of the Exponent class // ???is a static class and no instance
                                             // possible??? const One_minus_beta_times_n one_minus_beta_times_n;
                    const picongpu::float_64 factor(exponent(1.0 / (One_minus_beta_times_n()(n, particle))));
                    // factor=1/(1-beta*n)^g   g=2 for DFT and g=3 for FFT
                    return (n % ((n - beta) % beta_dot)) * factor;
                }
            };

            // typedef of all possible forms of Old_Method
            // typedef Old_Method<util::Cube<picongpu::float_64> > Old_FFT;
            typedef Old_Method<util::Square<picongpu::float_64>> Old_DFT;


            // ------- Calculate Amplitude class ------------- //

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

                HDINLINE Calc_Amplitude(
                    const Particle& particle,
                    const picongpu::float_64 delta_t,
                    const picongpu::float_64 t_sim)
                    : m_particle(particle)
                    , m_delta_t(delta_t)
                    , m_t_sim(t_sim)
                {
                }

                // get real vector part of amplitude

                HDINLINE vector_64 get_vector(const vector_64& n) const
                {
                    const vector_64 look_direction(n.unit_vec()); // make sure look_direction is a unit vector
                    VecCalc vecC;
                    return vecC(look_direction, m_particle, m_delta_t);
                }

                // get retarded time

                HDINLINE picongpu::float_64 get_t_ret(const vector_64 look_direction) const
                {
                    TimeCalc timeC;
                    return timeC(m_t_sim, look_direction, m_particle);

                    //  const vector_64 r = particle.get_location<When::now > (); // location
                    //  return (picongpu::float_64) (t - (n * r) / (picongpu::SPEED_OF_LIGHT));
                }

            private:
                // data:
                const Particle& m_particle; // one particle
                const picongpu::float_64 m_delta_t; // length of one time step in simulation
                const picongpu::float_64 m_t_sim; // simulation time (for methods not using index*delta_t )
            };

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
