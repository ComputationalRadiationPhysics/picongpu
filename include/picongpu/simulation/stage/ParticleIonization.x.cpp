/* Copyright 2013-2023 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#include "picongpu/simulation/stage/ParticleIonization.hpp"

#include "picongpu/particles/ParticlesFunctors.hpp"
#include "picongpu/particles/ionization/byCollision/collisionalIonizationCalc.hpp"
#include "picongpu/particles/ionization/byField/fieldIonizationCalc.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <boost/mpl/placeholders.hpp>

#include <cstdint>


namespace picongpu::simulation::stage
{
    void ParticleIonization::operator()(uint32_t const step) const
    {
        using pmacc::particles::traits::FilterByFlag;
        using SpeciesWithIonizers = typename FilterByFlag<VectorAllSpecies, ionizers<>>::type;
        pmacc::meta::ForEach<SpeciesWithIonizers, particles::CallIonization<boost::mpl::_1>> particleIonization;
        particleIonization(cellDescription, step);
    }
} // namespace picongpu::simulation::stage
