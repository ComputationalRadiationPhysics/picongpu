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

#include "picongpu/simulation/stage/Bremsstrahlung.hpp"

// Bremsstrahlung is available only with CUDA
#if(PMACC_CUDA_ENABLED == 1)

#    include "picongpu/particles/CallBremsstrahlung.tpp"
#    include "picongpu/particles/ParticlesFunctors.hpp"
#    include "picongpu/particles/bremsstrahlung/PhotonEmissionAngle.hpp"
#    include "picongpu/particles/bremsstrahlung/ScaledSpectrum.hpp"
#    include "picongpu/particles/bremsstrahlung/ScaledSpectrum.tpp"

#    include <pmacc/meta/ForEach.hpp>
#    include <pmacc/particles/traits/FilterByFlag.hpp>

#    include <cstdint>
#    include <map>
#endif

#include <stdexcept>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
#if(PMACC_CUDA_ENABLED == 1)
            class Bremsstrahlung::Impl
            {
            public:
                Impl(MappingDesc const cellDescription) : cellDescription(cellDescription)
                {
                    using AffectedSpecies = typename pmacc::particles::traits::
                        FilterByFlag<VectorAllSpecies, bremsstrahlungPhotons<>>::type;
                    // Initialize bremsstrahlung lookup tables, if there are species containing bremsstrahlung photons
                    if(!bmpl::empty<AffectedSpecies>::value)
                    {
                        meta::ForEach<AffectedSpecies, particles::bremsstrahlung::FillScaledSpectrumMap<bmpl::_1>>
                            fillScaledSpectrumMap;
                        fillScaledSpectrumMap(scaledSpectrumMap);
                        photonAngle.init();
                    }
                }

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

                using ScaledSpectrumMap = std::map<float_X, particles::bremsstrahlung::ScaledSpectrum>;

                //! Loopup table: atomic number -> scaled bremsstrahlung spectrum
                ScaledSpectrumMap scaledSpectrumMap;

                //! Loopup table for photon angle
                particles::bremsstrahlung::GetPhotonAngle photonAngle;
            };
#else
            //! Dummy implementation
            class Bremsstrahlung::Impl
            {
            public:
                Impl(MappingDesc const){};
                void operator()(uint32_t const step) const
                {
                }
            };
#endif

            Bremsstrahlung::Bremsstrahlung() = default;

            // Needed to be defined here, not in the .hpp, as Impl has to be a complete type
            Bremsstrahlung::~Bremsstrahlung() = default;

            void Bremsstrahlung::init(MappingDesc const cellDescription)
            {
                pImpl = std::make_unique<Impl>(cellDescription);
            }

            void Bremsstrahlung::operator()(uint32_t const step) const
            {
                if(!pImpl)
                    throw std::runtime_error("simulation::stage::Bremsstrahlung used without init() called");
                pImpl->operator()(step);
            }

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
