/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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
#include <pmacc/types.hpp>
#include <pmacc/cuSTL/cursor/Cursor.hpp>
#include <pmacc/cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include <pmacc/cuSTL/cursor/compile-time/SafeCursor.hpp>

#include "picongpu/fields/currentDeposition/Esirkepov/Esirkepov.def"
#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"
#include "picongpu/fields/currentDeposition/RelayPoint.hpp"


namespace picongpu
{
    namespace currentSolver
    {
        template<typename T_ParticleShape, typename T_Strategy>
        struct Esirkepov<T_ParticleShape, T_Strategy, DIM3>
        {
            using ParticleAssign = typename T_ParticleShape::ChargeAssignment;
            static constexpr int supp = ParticleAssign::support;

            static constexpr int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
            static constexpr int currentUpperMargin = (supp + 1) / 2 + 1;
            typedef pmacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
            typedef pmacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

            PMACC_CASSERT_MSG(
                __Esirkepov_supercell_or_number_of_guard_supercells_is_too_small_for_stencil,
                pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentLowerMargin
                    && pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentUpperMargin);

            float_X charge;

            /* At the moment Esirkepov only supports Yee cells where W is defined at origin (0,0,0)
             *
             * \todo: please fix me that we can use CenteredCell
             */
            template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType, typename T_Acc>
            DINLINE void operator()(
                T_Acc const& acc,
                DataBoxJ dataBoxJ,
                const PosType pos,
                const VelType velocity,
                const ChargeType charge,
                const float_X deltaTime)
            {
                this->charge = charge;
                const float3_X deltaPos = float3_X(
                    velocity.x() * deltaTime / cellSize.x(),
                    velocity.y() * deltaTime / cellSize.y(),
                    velocity.z() * deltaTime / cellSize.z());
                const PosType oldPos = pos - deltaPos;
                Line<float3_X> line(oldPos, pos);

                DataSpace<DIM3> gridShift;

                /* Define in which direction the particle leaves the cell.
                 * It is not relevant whether the particle leaves the cell via
                 * the positive or negative cell border.
                 *
                 * 0 == stay in cell
                 * 1 == leave cell
                 */
                DataSpace<simDim> leaveCell;

                /* calculate the offset for the virtual coordinate system */
                for(int d = 0; d < simDim; ++d)
                {
                    int iStart;
                    int iEnd;
                    constexpr bool isSupportEven = (supp % 2 == 0);
                    RelayPoint<isSupportEven>()(iStart, iEnd, line.m_pos0[d], line.m_pos1[d]);
                    gridShift[d] = iStart < iEnd ? iStart : iEnd; // integer min function
                    /* particle is leaving the cell */
                    leaveCell[d] = iStart != iEnd ? 1 : 0;
                    /* shift the particle position to the virtual coordinate system */
                    line.m_pos0[d] -= gridShift[d];
                    line.m_pos1[d] -= gridShift[d];
                }
                /* shift current field to the virtual coordinate system */
                auto cursorJ = dataBoxJ.shift(gridShift).toCursor();
                /**
                 * \brief the following three calls separate the 3D current deposition
                 * into three independent 1D calls, each for one direction and current component.
                 * Therefore the coordinate system has to be rotated so that the z-direction
                 * is always specific.
                 */
                using namespace cursor::tools;
                cptCurrent1D(
                    acc,
                    DataSpace<simDim>(leaveCell.y(), leaveCell.z(), leaveCell.x()),
                    twistVectorFieldAxes<pmacc::math::CT::Int<1, 2, 0>>(cursorJ),
                    rotateOrigin<1, 2, 0>(line),
                    cellSize.x());
                cptCurrent1D(
                    acc,
                    DataSpace<simDim>(leaveCell.z(), leaveCell.x(), leaveCell.y()),
                    twistVectorFieldAxes<pmacc::math::CT::Int<2, 0, 1>>(cursorJ),
                    rotateOrigin<2, 0, 1>(line),
                    cellSize.y());
                cptCurrent1D(acc, leaveCell, cursorJ, line, cellSize.z());
            }

            /**
             * deposites current in z-direction
             *
             * \param leaveCell vector with information if the particle is leaving the cell
             *         (for each direction, 0 means stays in cell and 1 means leaves cell)
             * \param cursorJ cursor pointing at the current density field of the particle's cell
             * \param line trajectory of the particle from to last to the current time step
             * \param cellEdgeLength length of edge of the cell in z-direction
             */
            template<typename CursorJ, typename T_Acc>
            DINLINE void cptCurrent1D(
                T_Acc const& acc,
                const DataSpace<simDim>& leaveCell,
                CursorJ cursorJ,
                const Line<float3_X>& line,
                const float_X cellEdgeLength)
            {
                /* skip calculation if the particle is not moving in z direction */
                if(line.m_pos0[2] == line.m_pos1[2])
                    return;

                constexpr int begin = -currentLowerMargin + 1;
                constexpr int end = begin + supp;

                /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
                 * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1)
                 */
                const float_X currentSurfaceDensity
                    = this->charge * (float_X(1.0) / float_X(CELL_VOLUME * DELTA_T)) * cellEdgeLength;

                /* pick every cell in the xy-plane that is overlapped by particle's
                 * form factor and deposit the current for the cells above and beneath
                 * that cell and for the cell itself.
                 *
                 * for loop optimization (help the compiler to generate better code):
                 *   - use a loop with a static range
                 *   - skip invalid indexes with a if condition around the full loop body
                 *     ( this helps the compiler to mask threads without work )
                 */
                for(int i = begin; i < end + 1; ++i)
                    if(i < end + leaveCell[0])
                    {
                        const float_X s0i = S0(line, i, 0);
                        const float_X dsi = S1(line, i, 0) - s0i;
                        for(int j = begin; j < end + 1; ++j)
                            if(j < end + leaveCell[1])
                            {
                                const float_X s0j = S0(line, j, 1);
                                const float_X dsj = S1(line, j, 1) - s0j;

                                float_X tmp = -currentSurfaceDensity
                                    * (s0i * s0j + float_X(0.5) * (dsi * s0j + s0i * dsj)
                                       + (float_X(1.0) / float_X(3.0)) * dsj * dsi);

                                float_X accumulated_J = float_X(0.0);

                                /* attention: inner loop has no upper bound `end + 1` because
                                 * the current for the point `end` is always zero,
                                 * therefore we skip the calculation
                                 */
                                for(int k = begin; k < end; ++k)
                                    if(k < end + leaveCell[2] - 1)
                                    {
                                        /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2)
                                         * version from Esirkepov paper. All coordinates are rotated before thus we can
                                         * always use C style W(i,j,k,2).
                                         */
                                        const float_X W = DS(line, k, 2) * tmp;
                                        accumulated_J += W;
                                        auto const atomicOp = typename T_Strategy::BlockReductionOp{};
                                        atomicOp(acc, (*cursorJ(i, j, k)).z(), accumulated_J);
                                    }
                            }
                    }
            }

            /** calculate S0 (see paper)
             * @param line element with previous and current position of the particle
             * @param gridPoint used grid point to evaluate assignment shape
             * @param d dimension range {0,1,2} means {x,y,z}
             *          different to Esirkepov paper, here we use C style
             */
            DINLINE float_X S0(const Line<float3_X>& line, const float_X gridPoint, const uint32_t d)
            {
                return ParticleAssign()(gridPoint - line.m_pos0[d]);
            }

            /** calculate S1 (see paper)
             * @param line element with previous and current position of the particle
             * @param gridPoint used grid point to evaluate assignment shape
             * @param d dimension range {0,1,2} means {x,y,z}
             *          different to Esirkepov paper, here we use C style
             */
            DINLINE float_X S1(const Line<float3_X>& line, const float_X gridPoint, const uint32_t d)
            {
                return ParticleAssign()(gridPoint - line.m_pos1[d]);
            }

            /** calculate DS (see paper)
             * @param line element with previous and current position of the particle
             * @param gridPoint used grid point to evaluate assignment shape
             * @param d dimension range {0,1,2} means {x,y,z}]
             *          different to Esirkepov paper, here we use C style
             */
            DINLINE float_X DS(const Line<float3_X>& line, const float_X gridPoint, const uint32_t d)
            {
                return ParticleAssign()(gridPoint - line.m_pos1[d]) - ParticleAssign()(gridPoint - line.m_pos0[d]);
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "Esirkepov");
                return propList;
            }
        };

    } // namespace currentSolver

} // namespace picongpu

#include "picongpu/fields/currentDeposition/Esirkepov/Esirkepov2D.hpp"
