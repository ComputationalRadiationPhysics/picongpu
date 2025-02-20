/* Copyright 2023-2024 Tapish Narwal
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
#include "picongpu/simulation_types.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/particles/Identifier.hpp>

#include <cstdint>

namespace picongpu
{
    namespace plugins::binning
    {
        enum class BinningType
        {
            Field,
            Particle
        };

        /**
         * @brief Provides knowledge of the simulation domain to the user
         * Names and concept are described at
         * https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
         */
        class DomainInfoBase
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
            HDINLINE DomainInfoBase(
                uint32_t simStep,
                pmacc::DataSpace<simDim> gOffset,
                pmacc::DataSpace<simDim> lOffset,
                pmacc::DataSpace<simDim> physicalSuperCellIdx)
                : currentStep{simStep}
                , globalOffset{gOffset}
                , localOffset{lOffset}
            {
                blockCellOffset = physicalSuperCellIdx * SuperCellSize::toRT();
            }
        };

        template<BinningType T_Binning>
        class DomainInfo;

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

        enum class PositionPrecision
        {
            // Returns the cell index in which the particle
            Cell,
            // Returns the position of the particle as the cell index + the position of the particle inside the cell
            // [0,1) This value is a floating point number of cells
            SubCell
        };

        enum class PositionUnits
        {
            // returns the position in SI units
            SI,
            // Returns the position as the number of cells (Integral value if PositionPrecision is Cell and
            // floating point if PositionPrecision is SubCell)
            Cell
        };

        template<>
        class DomainInfo<BinningType::Field> : public DomainInfoBase
        {
        public:
            pmacc::DataSpace<simDim> localCellIdx;
            HDINLINE DomainInfo(
                uint32_t simStep,
                pmacc::DataSpace<simDim> gOffset,
                pmacc::DataSpace<simDim> lOffset,
                pmacc::DataSpace<simDim> physicalSuperCellIdx,
                pmacc::DataSpace<simDim> localCellIndex)
                : DomainInfoBase(simStep, gOffset, lOffset, physicalSuperCellIdx)
                , localCellIdx{localCellIndex}
            {
            }

            // returns the cell position
            // passed Can also return in SI units if CellUnits::SI is specified
            template<DomainOrigin T_Origin, PositionUnits T_Units = PositionUnits::Cell>
            ALPAKA_FN_ACC auto getCellPosition() const
            {
                auto relative_cellpos = blockCellOffset;

                if constexpr(T_Origin == DomainOrigin::GLOBAL)
                {
                    relative_cellpos = relative_cellpos + localOffset;
                }
                if constexpr(T_Origin == DomainOrigin::TOTAL)
                {
                    relative_cellpos = relative_cellpos + globalOffset;
                }

                auto pos = localCellIdx + relative_cellpos;

                if constexpr(T_Units == PositionUnits::SI)
                {
                    return precisionCast<typename std::decay_t<decltype(sim.pic.getCellSize())>::type>(pos)
                        * sim.pic.getCellSize().shrink<simDim>();
                }

                return pos;
            }
        };

        template<>
        class DomainInfo<BinningType::Particle> : public DomainInfoBase
        {
        public:
            HDINLINE DomainInfo(
                uint32_t simStep,
                pmacc::DataSpace<simDim> gOffset,
                pmacc::DataSpace<simDim> lOffset,
                pmacc::DataSpace<simDim> physicalSuperCellIdx)
                : DomainInfoBase(simStep, gOffset, lOffset, physicalSuperCellIdx)
            {
            }
        };

        // returns the particle position
        // by default returns the cell index of the cell the particle is in
        // Can also return a fractional cell index representing the in cell position if PositionPrecision::SubCell is
        // passed Can also return in SI units if CellUnits::SI is specified
        template<
            DomainOrigin T_Origin,
            PositionPrecision T_Precision = PositionPrecision::Cell,
            PositionUnits T_Units = PositionUnits::Cell>
        ALPAKA_FN_ACC auto getParticlePosition(
            DomainInfo<BinningType::Particle> const& domainInfo,
            auto const& particle) -> pmacc::DataSpace<simDim>
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

            if constexpr(T_Precision == PositionPrecision::SubCell)
            {
                auto pos = precisionCast<typename std::decay_t<decltype(particle)>::type>(cellIdx + relative_cellpos)
                    + particle[position_];
                if constexpr(T_Units == PositionUnits::SI)
                {
                    return precisionCast<typename std::decay_t<decltype(particle)>::type>(pos)
                        * sim.pic.getCellSize().shrink<simDim>();
                }
                return pos;
            }
            else
            {
                auto pos = cellIdx + relative_cellpos;
                if constexpr(T_Units == PositionUnits::SI)
                {
                    return precisionCast<typename std::decay_t<decltype(particle)>::type>(pos) * sim.pic.getCellSize();
                }
                return pos;
            }
        }
    } // namespace plugins::binning
} // namespace picongpu
