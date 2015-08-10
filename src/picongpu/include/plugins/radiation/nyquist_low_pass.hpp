/**
 * Copyright 2013, 2015 Heiko Burau, Rene Widera, Richard Pausch
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
#include "parameters.hpp"
#include "particle.hpp"

class NyquistLowPass : public One_minus_beta_times_n
{

public:
    /**
     * calculates omega_{Nyquist} for particle in a direction n
     * omega_{Nyquist} = (\pi - \epsilon )/(\delta t * (1 - \vec(\beta) * \vec(n)))
     * so that all Amplitudes for higher frequencies can be ignored
    **/
    __device__ __host__ __forceinline__ NyquistLowPass(const vector_64& n, const Particle& particle)
      : omegaNyquist((picongpu::PI - 0.01)/
           (picongpu::DELTA_T *
            One_minus_beta_times_n()(n, particle)))
    { }

    /**
     * default constructor - needed for allocating shared memory on GPU (Radiation.hpp kernel)
    **/
    __device__ __host__ __forceinline__ NyquistLowPass(void)
    { }


    /**
     * checks if frequency omega is below Nyquist frequency
    **/
    __device__ __host__ __forceinline__ bool check(const picongpu::float_32 omega)
    {
        return omega < omegaNyquist * picongpu::radiationNyquist::NyquistFactor;
    }

private:
    picongpu::float_32 omegaNyquist; // Nyquist frequency for a particle (at a certain time step) for one direction
};

