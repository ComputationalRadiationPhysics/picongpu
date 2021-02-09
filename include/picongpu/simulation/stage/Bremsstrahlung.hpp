/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
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

// Bremsstrahlung is available only with CUDA
#if(PMACC_CUDA_ENABLED == 1)

#    include "picongpu/particles/bremsstrahlung/PhotonEmissionAngle.hpp"
#    include "picongpu/particles/bremsstrahlung/ScaledSpectrum.hpp"
#    include "picongpu/particles/ParticlesFunctors.hpp"

#    include <pmacc/meta/ForEach.hpp>
#    include <pmacc/particles/traits/FilterByFlag.hpp>

#    include <cstdint>
#    include <map>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop computing Bremsstrahlung
             *
             * Only affects particle species with the bremsstrahlungIons attribute.
             */
            class Bremsstrahlung
            {
            public:
                using ScaledSpectrumMap = std::map<float_X, particles::bremsstrahlung::ScaledSpectrum>;

                /** Create a Bremsstrahlung functor
                 *
                 * Having this in constructor is a temporary solution.
                 *
                 * @param cellDescription mapping for kernels
                 * @param scaledSpectrumMap initialized spectrum lookup table
                 * @param photonAngle initialized photon angle lookup table
                 */
                Bremsstrahlung(
                    MappingDesc const cellDescription,
                    ScaledSpectrumMap& scaledSpectrumMap,
                    particles::bremsstrahlung::GetPhotonAngle& photonAngle)
                    : cellDescription(cellDescription)
                    , scaledSpectrumMap(scaledSpectrumMap)
                    , photonAngle(photonAngle)
                {
                }

                /** Ionize particles
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const step) const
                {
                    using pmacc::particles::traits::FilterByFlag;
                    using SpeciesWithBremsstrahlung =
                        typename FilterByFlag<VectorAllSpecies, bremsstrahlungIons<>>::type;
                    pmacc::meta::ForEach<SpeciesWithBremsstrahlung, particles::CallBremsstrahlung<bmpl::_1>>
                        particleBremsstrahlung;
                    particleBremsstrahlung(cellDescription, step, scaledSpectrumMap, photonAngle);
                }

            private:
                //! Mapping for kernels
                MappingDesc cellDescription;

                //! Loopup table: atomic number -> scaled bremsstrahlung spectrum
                ScaledSpectrumMap& scaledSpectrumMap;

                //! Loopup table for photon angle
                particles::bremsstrahlung::GetPhotonAngle& photonAngle;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu

#endif // ( PMACC_CUDA_ENABLED == 1 )
