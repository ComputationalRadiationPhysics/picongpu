/* Copyright 2023 Tapish Narwal
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

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>

namespace picongpu
{
    namespace plugins::binning
    {
        /**
         * @brief Provides knowledge of the simulation domain to the user
         * Names and concept are described at
         * https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
         */
        class DomainInfo
        {
        public:
            /** Current simulation timestep */
            uint32_t currentStep;
            /** Offset of the global domain on all GPUs */
            pmacc::DataSpace<simDim> globalOffset;
            /** Offset of the domain simulated on current GPU */
            pmacc::DataSpace<simDim> localOffset;
            /** Offset of domain simulated by current block wrt the border */
            pmacc::DataSpace<simDim> blockCellOffset;

            /**
             * @param physicalSuperCellIdx supercell index relative to the border origin
             */
            DINLINE DomainInfo(
                uint32_t simStep,
                pmacc::DataSpace<simDim> gOffset,
                pmacc::DataSpace<simDim> lOffset,
                DataSpace<simDim> physicalSuperCellIdx)
                : currentStep{simStep}
                , globalOffset{gOffset}
                , localOffset{lOffset}
            {
                blockCellOffset = physicalSuperCellIdx * SuperCellSize::toRT();
            }
        };

        enum class DomainOrigin
        {
            // absolute origin of the simulation, inlcudes areas that are not in the current global volume,
            // i.e. areas that have gone out due to the sliding window
            TOTAL,
            // origin of the current sliding window, i.e. the currently simulated volume over all GPUs, no guards
            GLOBAL,
            // origin of the current ("my") GPU, no guards
            LOCAL
        };

        template<DomainOrigin T_Origin, typename T_Particle>
        ALPAKA_FN_ACC auto getParticlePosition(DomainInfo domainInfo, T_Particle particle) -> pmacc::DataSpace<simDim>
        {
            int const linearCellIdx = particle[localCellIdx_];
            DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
            auto relative_cellpos = domainInfo.blockCellOffset;

            if constexpr(T_Origin == DomainOrigin::GLOBAL)
            {
                relative_cellpos = relative_cellpos + domainInfo.localOffset;
            }
            if constexpr(T_Origin == DomainOrigin::TOTAL)
            {
                relative_cellpos = relative_cellpos + domainInfo.globalOffset;
            }

            auto posBin = cellIdx + relative_cellpos;
            return posBin;
        }
    } // namespace plugins::binning
} // namespace picongpu
