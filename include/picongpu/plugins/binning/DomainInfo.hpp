/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

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
#include "picongpu/param/dimension.param"
#include "picongpu/simulation/control/MovingWindow.hpp"
#include "picongpu/simulation_types.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/vector/Vector.hpp>
#include <pmacc/particles/Identifier.hpp>

#include <algorithm>
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
         *
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
            /** Size of the guard region around the local domain in cells */
            pmacc::DataSpace<simDim> guardSize;
            /** Moving window offset accurate to sub-cell position */
            pmacc::math::Vector<float_64, simDim> windowOffset;

            /**
             * @param physicalSuperCellIdx supercell index relative to the border origin
             */
            HDINLINE DomainInfoBase(
                uint32_t simStep,
                pmacc::DataSpace<simDim> gOffset,
                pmacc::DataSpace<simDim> lOffset,
                pmacc::DataSpace<simDim> guardingSuperCells,
                pmacc::math::Vector<float_64, simDim> wOffset)
                : currentStep{simStep}
                , globalOffset{gOffset}
                , localOffset{lOffset}
                , guardSize{guardingSuperCells * SuperCellSize::toRT()}
                , windowOffset{wOffset}
            {
            }

            DINLINE void fillDeviceData(pmacc::DataSpace<simDim> physicalSuperCellIdx)
            {
                blockCellOffset = physicalSuperCellIdx * SuperCellSize::toRT();
            }
        };

        template<BinningType T_Binning>
        class DomainInfo;

        enum class DomainOrigin
        {
            /** Absolute origin of the simulation. This includes regions that are no longer part of the current global
             * volume because they have moved out of the sliding window. */
            TOTAL,
            /** Origin of the current sliding window, i.e. the currently simulated domain across all GPUs, excluding
             * guard cells. */
            GLOBAL,
            /** Origin of the local domain on the current GPU, excluding guard cells. */
            LOCAL,
            /** Origin relative to the origin of the sliding window. This origin only starts moving with the sliding
             *  window and is not discretized to the cell grid
             */
            MOVING_WINDOW,
            /** Origin of the local domain on the current GPU, including guard cells. This setting is in particular
             * used to access field data for the current cell with getCellIndex. */
            LOCAL_WITH_GUARDS
        };

        enum class PositionPrecision
        {
            /** Returns the particle position at cell precision, i.e. as a cell index. */
            CELL,
            /** Returns the particle position with sub-cell precision, i.e. as the cell index plus the particle
             * position inside the cell in the range [0,1). The result is therefore a floating-point position in units
             * of cells. */
            SUB_CELL
        };

        /**
         * @brief Output unit type for particle position.
         */
        enum class PositionUnits
        {
            /**
             * @brief Returns the position in SI units.
             * @note Converting the particle positions to SI might be dangerous, especially with respect to the total
             * origin, as floating point numbers lose precision as the distance from the origin increases.
             */
            SI,
            /**
             * @brief Returns the position in PIC units.
             * @note Converting the particle positions to PIC might be dangerous, especially with respect to the total
             * origin, as floating point numbers lose precision as the distance from the origin increases.
             */
            PIC,
            /**
             * @brief Returns the position as the number of cells.
             * Integral value if PositionPrecision is Cell and floating point if PositionPrecision is SubCell.
             */
            CELL
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
                pmacc::DataSpace<simDim> guardingSuperCells,
                pmacc::math::Vector<float_64, simDim> windowOffset)
                : DomainInfoBase(simStep, gOffset, lOffset, guardingSuperCells, windowOffset)
            {
            }

            DINLINE void fillDeviceData(
                pmacc::DataSpace<simDim> physicalSuperCellIdx,
                pmacc::DataSpace<simDim> localCellIndex)
            {
                localCellIdx = localCellIndex;
                DomainInfoBase::fillDeviceData(physicalSuperCellIdx);
            }

            // returns the cell index. To get the exact position of the fields, use the fieldPosition trait
            // passed Can also return in SI units if CellUnits::SI is specified
            template<DomainOrigin T_Origin, PositionUnits T_Units = PositionUnits::CELL>
            ALPAKA_FN_ACC auto getCellIndex() const
            {
                auto relative_cellpos = blockCellOffset;

                if constexpr(T_Origin == DomainOrigin::LOCAL_WITH_GUARDS)
                {
                    relative_cellpos = relative_cellpos + guardSize;
                }
                if constexpr(T_Origin == DomainOrigin::GLOBAL)
                {
                    relative_cellpos = relative_cellpos + localOffset;
                }
                if constexpr(T_Origin == DomainOrigin::TOTAL)
                {
                    relative_cellpos = relative_cellpos + localOffset + globalOffset;
                }

                auto pos = localCellIdx + relative_cellpos;

                using DistanceReturnType = std::common_type_t<
                    typename std::decay_t<decltype(sim.si.getCellSize())>::type,
                    typename std::decay_t<decltype(pos)>::type>;

                if constexpr(T_Units == PositionUnits::SI)
                {
                    return precisionCast<DistanceReturnType>(pos)
                           * precisionCast<DistanceReturnType>(sim.si.getCellSize().shrink<simDim>());
                }
                else if constexpr(T_Units == PositionUnits::PIC)
                {
                    return precisionCast<DistanceReturnType>(pos)
                           * precisionCast<DistanceReturnType>(sim.pic.getCellSize().shrink<simDim>());
                }
                // else T_Units == PositionUnits::Cell
                else
                {
                    return pos;
                }
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
                pmacc::DataSpace<simDim> guardingSuperCells,
                pmacc::math::Vector<float_64, simDim> windowOffset)
                : DomainInfoBase(simStep, gOffset, lOffset, guardingSuperCells, windowOffset)
            {
            }

            DINLINE void fillDeviceData(pmacc::DataSpace<simDim> physicalSuperCellIdx)
            {
                DomainInfoBase::fillDeviceData(physicalSuperCellIdx);
            }
        };

        namespace concepts
        {
            template<auto V, auto... Vs>
            concept OneOf = ((V == Vs) || ...);

            template<DomainOrigin Origin, PositionPrecision Precision>
            concept ValidPositionRequest
                = OneOf<
                      Origin,
                      DomainOrigin::TOTAL,
                      DomainOrigin::LOCAL,
                      DomainOrigin::GLOBAL,
                      DomainOrigin::LOCAL_WITH_GUARDS>
                  || (Origin == DomainOrigin::MOVING_WINDOW && Precision == PositionPrecision::SUB_CELL);
        } // namespace concepts

        /**
         * @brief Returns the particle position as a pmacc vector.
         *
         * By default, returns the cell index of the cell the particle is in.
         * Returns a fractional cell index representing the in-cell position if PositionPrecision::SubCell is passed.
         * Returns in SI units if PositionUnits::SI is specified.
         *
         * @warning Converting the particle positions to SI might be dangerous, especially with respect to the total
         * origin, as floating point numbers lose precision as the distance from the origin increases.
         *
         * @warning The comoving frame defined by DomainOrigin::MOVING_WINDOW is only supported with SUBCELL precision.
         *
         * @tparam T_Origin The origin reference for the position.
         * @tparam T_Precision The precision of the position (CELL index or SUBCELL position).
         * @tparam T_Units The units of the position (SI, PIC or CELL).
         * @param domainInfo The domain information.
         * @param particle The particle whose position is to be determined.
         * @return The particle position as a pmacc vector.
         */
        template<
            DomainOrigin T_Origin,
            PositionPrecision T_Precision = PositionPrecision::CELL,
            PositionUnits T_Units = PositionUnits::CELL>
        requires concepts::ValidPositionRequest<T_Origin, T_Precision>
        ALPAKA_FN_ACC auto getParticlePosition(
            DomainInfo<BinningType::Particle> const& domainInfo,
            auto const& particle)
        {
            int const linearCellIdx = particle[localCellIdx_];
            pmacc::DataSpace<simDim> const cellIdx = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
            auto relative_cellpos = domainInfo.blockCellOffset + cellIdx;

            if constexpr(T_Origin == DomainOrigin::LOCAL_WITH_GUARDS)
            {
                relative_cellpos = relative_cellpos + domainInfo.guardSize;
            }
            if constexpr(T_Origin == DomainOrigin::GLOBAL)
            {
                relative_cellpos = relative_cellpos + domainInfo.localOffset;
            }
            if constexpr(T_Origin == DomainOrigin::TOTAL)
            {
                relative_cellpos = relative_cellpos + domainInfo.localOffset + domainInfo.globalOffset;
            }
            // treat as total and in the sub cell calculation, subtract the origin location
            if constexpr(T_Origin == DomainOrigin::MOVING_WINDOW)
            {
                relative_cellpos = relative_cellpos + domainInfo.localOffset + domainInfo.globalOffset;
            }
            if constexpr(T_Precision == PositionPrecision::SUB_CELL)
            {
                using DistanceReturnType = std::common_type_t<
                    typename std::decay_t<decltype(sim.si.getCellSize())>::type,
                    typename std::decay_t<decltype(particle[position_])>::type>;

                auto pos = precisionCast<DistanceReturnType>(relative_cellpos)
                           + precisionCast<DistanceReturnType>(particle[position_]);

                if constexpr(T_Origin == DomainOrigin::MOVING_WINDOW)
                {
                    pos = pos - precisionCast<DistanceReturnType>(domainInfo.windowOffset);
                }

                if constexpr(T_Units == PositionUnits::SI)
                {
                    auto cellSize = sim.si.getCellSize().shrink<simDim>();
                    return precisionCast<DistanceReturnType>(pos) * precisionCast<DistanceReturnType>(cellSize);
                }
                else if constexpr(T_Units == PositionUnits::PIC)
                {
                    auto cellSize = sim.pic.getCellSize().shrink<simDim>();
                    return precisionCast<DistanceReturnType>(pos) * precisionCast<DistanceReturnType>(cellSize);
                }
                // else T_Units == PositionUnits::Cell
                else
                {
                    return pos;
                }
            }
            // T_Precision == PositionPrecision::Cell
            else
            {
                using DistanceReturnType = std::common_type_t<
                    typename std::decay_t<decltype(sim.si.getCellSize())>::type,
                    typename std::decay_t<decltype(relative_cellpos)>::type>;

                if constexpr(T_Units == PositionUnits::SI)
                {
                    auto cellSize = sim.si.getCellSize().shrink<simDim>();
                    return precisionCast<DistanceReturnType>(relative_cellpos)
                           * precisionCast<DistanceReturnType>(cellSize);
                }
                else if constexpr(T_Units == PositionUnits::PIC)
                {
                    auto cellSize = sim.pic.getCellSize().shrink<simDim>();
                    return precisionCast<DistanceReturnType>(relative_cellpos)
                           * precisionCast<DistanceReturnType>(cellSize);
                }
                // else T_Units == PositionUnits::Cell
                else
                {
                    return relative_cellpos;
                }
            }
        }
    } // namespace plugins::binning
} // namespace picongpu
