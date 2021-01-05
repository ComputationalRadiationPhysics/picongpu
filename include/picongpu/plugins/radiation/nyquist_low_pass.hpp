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

#include "calc_amplitude.hpp"
#include "VectorTypes.hpp"
#include "particle.hpp"


namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            class NyquistLowPass : public One_minus_beta_times_n
            {
            public:
                /**
                 * calculates \f$omega_{Nyquist}\f$ for particle in a direction \f$n\f$
                 * \f$omega_{Nyquist} = (\pi - \epsilon )/(\delta t * (1 - \vec(\beta) * \vec(n)))\f$
                 * so that all Amplitudes for higher frequencies can be ignored
                 **/
                HDINLINE NyquistLowPass(const vector_64& n, const Particle& particle)
                    : omegaNyquist((PI - 0.01) / (DELTA_T * One_minus_beta_times_n()(n, particle)))
                {
                }

                /**
                 * default constructor - needed for allocating shared memory on GPU (Radiation.hpp kernel)
                 **/
                HDINLINE NyquistLowPass(void)
                {
                }


                /**
                 * checks if frequency omega is below Nyquist frequency
                 **/
                HDINLINE bool check(const float_32 omega)
                {
                    return omega < omegaNyquist * radiationNyquist::NyquistFactor;
                }

            private:
                float_32 omegaNyquist; // Nyquist frequency for a particle (at a certain time step) for one direction
            };

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
