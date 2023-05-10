/* Copyright 2023 Brian Marre
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

/** @file implements a T_PositionFunctor that places numParticlesPerSuperCell macro particles at a
 *  fixed location in each super cell with weighting according to the local density
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/startPosition/OneSuperCellPositionImpl.def"
#include "picongpu/particles/startPosition/detail/WeightMacroParticles.hpp"

#include <pmacc/traits/HasIdentifier.hpp>

namespace picongpu::particles::startPosition::acc
{
    template<typename T_ParamClass>
    struct OneSuperCellPositionImpl
    {
        /** set in-superCell position and weighting
         *
         * @tparam T_Particle pmacc::Particle, particle type
         * @tparam T_Args pmacc::Particle, arbitrary number of particles types
         *
         * @param particle particle to be manipulated
         * @param ... init input particles particles, unused here
         */
        template<typename T_Particle, typename... T_Args>
        HDINLINE void operator()(T_Particle& particle, T_Args&&...)
        {
            particle[position_] = T_ParamClass{}.inCellOffset.template shrink<simDim>();

            // set the weighting attribute if the particle species has it
            constexpr bool hasWeighting
                = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, weighting>::type::value;
            if constexpr(hasWeighting)
                particle[weighting_] = m_weighting;
        }

        /** predictor for number of macro particles to create in a cell
         *
         * @attention expects realParticlesPerCell to be per superCell
         */
        template<typename T_Particle>
        HDINLINE uint32_t
        numberOfMacroParticles(float_X const realParticlesPerCell, DataSpace<simDim> const cellIdx) const
        {
            PMACC_CASSERT_MSG(
                spawnCellIdx_dim_and_simDim_are_inconsistent,
                T_ParamClass::spawnCellIdx::dim != picongpu::simDim);

            auto spawnCellIdx = T_ParamClass::spawnCellIdx::toRT();
            auto superCellSize = picongpu::SuperCellSize::toRT();

            // check for invalid spawnCellIdx
            for(uint8_t i = static_cast<uint8_t>(0u); i < picongpu::simDim; i++)
            {
                if(spawnCellIdx[i] >= superCellSize[i])
                    return static_cast<uint32_t>(0u);
            }

            // is cellIdx == spawnCellIdx?
            bool cellIsSpawnCell = true;
            for(uint8_t i = static_cast<uint8_t>(0u); i < picongpu::simDim; i++)
            {
                if(spawnCellIdx[i] != cellIdx[i])
                {
                    cellIsSpawnCell = false;
                }
            }

            if(!cellIsSpawnCell)
                return static_cast<uint32_t>(0u);

            /// @attention m_weighting member might stay uninitialized!
            uint32_t result = T_ParamClass::numParticlesPerSuperCell;

            constexpr bool hasWeighting
                = pmacc::traits::HasIdentifier<typename T_Particle::FrameType, weighting>::type::value;

            if constexpr(hasWeighting)
                result = startPosition::detail::WeightMacroParticles{}(
                    realParticlesPerCell,
                    T_ParamClass::numParticlesPerSuperCell,
                    m_weighting);

            return result;
        }

    private:
        //! position dependent weighting init
        float_X m_weighting;
    };
} // namespace picongpu::particles::startPosition::acc
