/*
 * SPDX-FileCopyrightText: Heiko Burau, Rene Widera, Richard Pausch, Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/plugins/radiation/VectorTypes.hpp"
#include "picongpu/plugins/radiation/calc_amplitude.hpp"
#include "picongpu/plugins/radiation/param.hpp"
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
                HDINLINE NyquistLowPass(vector_64 const& n, Particle const& particle)
                {
                    auto const omegaNyquist = (PI - 0.01) / (sim.pic.getDt() * OneMinusBetaTimesN()(n, particle));
                    threshold = static_cast<float_X>(omegaNyquist * radiationNyquist::NyquistFactor);
                }

                //! Default constructor - needed for allocating shared memory on GPU (Radiation.kernel)
                HDINLINE NyquistLowPass() = default;

                //! Checks if frequency omega is below the threshold
                HDINLINE bool check(float_X const omega) const
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
