/* Copyright 2020-2021 Pawel Ordyna
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
#include <pmacc/traits/HasFlag.hpp>
#include "picongpu/particles/particleToGrid/derivedAttributes/DerivedAttributes.def"

namespace picongpu
{
    namespace plugins
    {
        namespace xrayScattering
        {
            using namespace particles::particleToGrid;

            template<typename T_ParticleType>
            struct IsIon
            {
                using FrameType = typename T_ParticleType::FrameType;
                using type = typename pmacc::traits::HasFlag<FrameType, boundElectrons>::type;
            };


            /** Chose an electron density solver for a given particle type.
             *
             * Switches between a bound electron number density solver for particles
             * with the boundElectrons attribute (ions) and a particle number density
             * solver for other particle types (electrons).
             *
             * @tparam T_ParticleType Scattering particles
             * @return ::type TmpField solver to be used
             */
            template<typename T_ParticlesType>
            struct DetermineElectronDensitySolver
            {
                using IonSolver =
                    typename CreateFieldTmpOperation_t<T_ParticlesType, derivedAttributes::BoundElectronDensity>::
                        Solver;

                using ElectronSolver =
                    typename CreateFieldTmpOperation_t<T_ParticlesType, derivedAttributes::Density>::Solver;

                using type =
                    typename boost::mpl::if_<typename IsIon<T_ParticlesType>::type, IonSolver, ElectronSolver>::type;
            };
        } // namespace xrayScattering
    } // namespace plugins
} // namespace picongpu
