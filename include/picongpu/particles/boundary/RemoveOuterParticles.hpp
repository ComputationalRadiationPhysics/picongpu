/* Copyright 2021 Sergei Bastrakov
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

#include "picongpu/particles/boundary/Absorbing.hpp"
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffset.hpp"

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            /** Remove outer particles of the given species
             *
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param currentStep current time iteration
             */
            template<typename T_Species>
            inline void removeOuterParticles(T_Species& species, uint32_t currentStep)
            {
                // Here we do basically same as in absorbing boundaries, but for the whole domain
                pmacc::DataSpace<simDim> beginInternalCellsTotal, endInternalCellsTotal;
                getInternalCellsTotal(species, &beginInternalCellsTotal, &endInternalCellsTotal);
                AbsorbParticleIfOutside::parameters().beginInternalCellsTotal = beginInternalCellsTotal;
                AbsorbParticleIfOutside::parameters().endInternalCellsTotal = endInternalCellsTotal;
                using Manipulator = manipulators::unary::FreeTotalCellOffset<AbsorbParticleIfOutside>;
                particles::manipulate<Manipulator, T_Species>(currentStep);
                // Fill gaps to finalize deletion
                species.fillAllGaps();
            }

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
