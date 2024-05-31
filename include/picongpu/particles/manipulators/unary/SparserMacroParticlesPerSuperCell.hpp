/* Copyright 2023-2024 Brian Marre, Richard Pausch
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

//! @file defines sparser functor for keeping only macro particles from one cell per superCell

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <cstdint>
#include <string_view>

namespace picongpu::particles::manipulators::unary::acc
{
    namespace detail
    {
        struct SparserParam
        {
            //! local index of cell in each superCell whose macro particles to keep
            using idxCellToKeep = pmacc::math::CT::shrinkTo<pmacc::math::CT::Int<1, 1, 1>, picongpu::simDim>::type;
        };
    } // namespace detail


    /** functor marking for deletion all passed macro particles which are not in a specified cell of a superCell and
     * updating the weight of the remaining particles to approximately keep the total density per superCell constant
     * under the assumption of homogenous density distribution within each superCell.
     *
     * to be passed as template parameter to picongpu::particles::manipulators::unary::FreeTotalCellOffset
     *
     * @tparam T_SparserParam param type setting which in superCell cellIdx to keep macro particles
     *
     * @attention any application of this functor to a species must be followed by FillAllGaps<SpeciesName> to
     *  preserve particle data integrity!
     */
    template<typename T_SparserParam = detail::SparserParam>
    struct SparserMacroParticlesPerSuperCell
    {
        template<typename T_Particle>
        HDINLINE void operator()(
            DataSpace<picongpu::simDim> const& particleOffsetInCellsToTotalOrigin,
            T_Particle& particle)
        {
            DataSpace<picongpu::simDim> const superCellSize_RT = picongpu::SuperCellSize::toRT();
            DataSpace<picongpu::simDim> const idxCellToKeep_RT = T_SparserParam::idxCellToKeep::toRT();

            // is particle in cell whose particles we want to keep?
            bool const keepParticle = ((particleOffsetInCellsToTotalOrigin % superCellSize_RT) == idxCellToKeep_RT);
            if(!keepParticle)
            {
                // mark particle as invalid
                particle[multiMask_] = 0;
                return;
            }
            else
            {
                constexpr bool hasWeighting
                    = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, weighting>::type::value;

                if constexpr(hasWeighting)
                    // update  weight to approximately maintain total density
                    particle[weighting_] *= pmacc::math::CT::volume<typename picongpu::SuperCellSize>::type::value;
            }
        }

        static constexpr char const* name = "SparserMacroParticlesPerSuperCell";
    };
} // namespace picongpu::particles::manipulators::unary::acc
