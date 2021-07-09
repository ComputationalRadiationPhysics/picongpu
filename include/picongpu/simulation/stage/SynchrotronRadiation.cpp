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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/simulation/stage/SynchrotronRadiation.hpp"

#include "picongpu/particles/synchrotronPhotons/SynchrotronFunctions.hpp"
#include "picongpu/particles/synchrotronPhotons/SynchrotronFunctions.tpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>
#include <memory>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            class SynchrotronRadiation::Impl
            {
            public:
                Impl(MappingDesc const cellDescription) : cellDescription(cellDescription)
                {
                    using AllSynchrotronPhotonsSpecies =
                        typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, synchrotronPhotons<>>::type;
                    // Initialize synchrotron functions, if there are synchrotron photon species
                    if(!bmpl::empty<AllSynchrotronPhotonsSpecies>::value)
                        functions.init();
                }

                void operator()(uint32_t const step) const
                {
                    using pmacc::particles::traits::FilterByFlag;
                    using SynchrotronPhotonsSpecies =
                        typename FilterByFlag<VectorAllSpecies, synchrotronPhotons<>>::type;
                    pmacc::meta::ForEach<SynchrotronPhotonsSpecies, particles::CallSynchrotronPhotons<bmpl::_1>>
                        synchrotronRadiation;
                    synchrotronRadiation(cellDescription, step, functions);
                }

            private:
                //! Mapping for kernels
                MappingDesc cellDescription;

                //! Initialized synchrotron functions
                particles::synchrotronPhotons::SynchrotronFunctions functions;
            };

            SynchrotronRadiation::SynchrotronRadiation() = default;

            // Needed to be defined here, not in the .hpp, as Impl has to be a complete type
            SynchrotronRadiation::~SynchrotronRadiation() = default;

            void SynchrotronRadiation::init(MappingDesc const cellDescription)
            {
                pImpl = std::make_unique<Impl>(cellDescription);
            }

            void SynchrotronRadiation::operator()(uint32_t const step) const
            {
                if(!pImpl)
                    throw std::runtime_error("simulation::stage::SynchrotronRadiation used without init() called");
                pImpl->operator()(step);
            }

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
