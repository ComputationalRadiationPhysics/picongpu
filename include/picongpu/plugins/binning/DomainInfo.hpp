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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace picongpu
{
    namespace plugins::binning
    {
        /**
         * @brief Provides knowledge of the simulation domain to the user
         */
        class DomainInfo
        {
        public:
            uint32_t currentStep; ///< Current simulation timestep
            pmacc::DataSpace<SIMDIM> globalOffset; ///< Offset of the global domain on all GPUs
            pmacc::DataSpace<SIMDIM> localOffset; ///< Offset of the domain simulated on current GPU
            pmacc::DataSpace<SIMDIM> blockCellOffset; ///< Offset of domain simulated by current block wrt the border

            /**
             * @param physicalSuperCellIdx supercell index relative to the border origin
             */
            DINLINE DomainInfo(
                uint32_t simStep,
                pmacc::DataSpace<SIMDIM> gOffset,
                pmacc::DataSpace<SIMDIM> lOffset,
                DataSpace<SIMDIM> physicalSuperCellIdx)
                : currentStep{simStep}
                , globalOffset{gOffset}
                , localOffset{lOffset}
            {
                blockCellOffset = physicalSuperCellIdx * SuperCellSize::toRT();
            }
        };
    } // namespace plugins::binning

} // namespace picongpu