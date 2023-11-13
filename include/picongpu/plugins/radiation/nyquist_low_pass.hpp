/* Copyright 2013-2023 Heiko Burau, Rene Widera, Richard Pausch, Sergei Bastrakov
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

#include "picongpu/plugins/radiation/VectorTypes.hpp"
#include "picongpu/plugins/radiation/calc_amplitude.hpp"
#include "picongpu/plugins/radiation/particle.hpp"


namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            //! Low pass filter on frequencies, the threshold depends on the Nyquist frequency
            class NyquistLowPass : public OneMinusBetaTimesN
            {
            public:
                /** Calculates the filter threshold, only frequencies below it pass
                 *
                 * The threshold is equal to \f$omega_{Nyquist} * NyquistFactor\f$ for particle in a direction \f$n\f$
                 * \f$omega_{Nyquist} = (\pi - \epsilon )/(\delta t * (1 - \vec(\beta) * \vec(n)))\f$
                 * so that all Amplitudes for higher frequencies can be ignored.
                 * The Nyquist factor value is set in radiation.param.
                 **/
                HDINLINE NyquistLowPass(const vector_64& n, const Particle& particle)
                {
                    auto const omegaNyquist = (PI - 0.01) / (DELTA_T * OneMinusBetaTimesN()(n, particle));
                    threshold = static_cast<float_X>(omegaNyquist * radiationNyquist::NyquistFactor);
                }

                //! Default constructor - needed for allocating shared memory on GPU (Radiation.kernel)
                HDINLINE NyquistLowPass() = default;

                //! Checks if frequency omega is below the threshold
                HDINLINE bool check(const float_X omega) const
                {
                    return omega < threshold;
                }

            private:
                // Nyquist frequency for a particle (at a certain time step) for one direction multiplied by the
                // Nyquist factor
                float_X threshold;
            };

        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
