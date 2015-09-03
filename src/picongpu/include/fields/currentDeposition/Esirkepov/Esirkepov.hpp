/**
 * Copyright 2013-2014 Axel Huebl, Heiko Burau, Rene Widera
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

#include "simulation_defines.hpp"
#include "types.h"
#include "cuSTL/cursor/Cursor.hpp"
#include "basicOperations.hpp"
#include <cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include <cuSTL/cursor/compile-time/SafeCursor.hpp>
#include "fields/currentDeposition/Esirkepov/Esirkepov.def"
#include "fields/currentDeposition/Esirkepov/Line.hpp"

namespace picongpu
{
namespace currentSolver
{
using namespace PMacc;

template<typename T_ParticleShape>
struct Esirkepov<T_ParticleShape, DIM3>
{
    typedef typename T_ParticleShape::ChargeAssignment ParticleAssign;
    static const int supp = ParticleAssign::support;

    static const int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
    static const int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef PMacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
    typedef PMacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

    /* begin and end border is calculated for the current time step were the old
     * position of the particle in the previous time step is smaller than the current position
     * Later on all coordinates are shifted thus we can solve the charge calculation
     * in support + 1 steps.
     *
     * For the case were previous position is greater than current position we correct
     * begin and end on runtime and add +1 to begin and end.
     */
    static const int begin = -currentLowerMargin;
    static const int end = begin + supp + 1;

    float_X charge;

    /* At the moment Esirkepov only support YeeCell were W is defined at origin (0,0,0)
     *
     * \todo: please fix me that we can use CenteredCell
     */
    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos,
                            const VelType velocity,
                            const ChargeType charge,
                            const float_X deltaTime)
    {
        this->charge = charge;
        const float3_X deltaPos = float3_X(velocity.x() * deltaTime / cellSize.x(),
                                           velocity.y() * deltaTime / cellSize.y(),
                                           velocity.z() * deltaTime / cellSize.z());
        const PosType oldPos = pos - deltaPos;
        Line<float3_X> line(oldPos, pos);
        BOOST_AUTO(cursorJ, dataBoxJ.toCursor());

        if (supp % 2 == 1)
        {
            /* odd support
             * shift coordinate system that we always can solve Esirkepov by going
             * over the grid points [begin,end)
             */

            /* for any direction
             * if pos> 0.5
             * shift curser+1 and new_pos=old_pos-1
             *
             * floor(pos*2.0) is equal (pos > 0.5)
             */
            float3_X coordinate_shift(
                                      float_X(math::floor(pos.x() * float_X(2.0))),
                                      float_X(math::floor(pos.y() * float_X(2.0))),
                                      float_X(math::floor(pos.z() * float_X(2.0)))
                                      );
            cursorJ = cursorJ(
                              PMacc::math::Int < 3 > (
                                                      coordinate_shift.x(),
                                                      coordinate_shift.y(),
                                                      coordinate_shift.z()
                                                      ));
            //same as: pos = pos - coordinate_shift;
            line.m_pos0 -= (coordinate_shift);
            line.m_pos1 -= (coordinate_shift);
        }

        /**
         * \brief the following three calls separate the 3D current deposition
         * into three independent 1D calls, each for one direction and current component.
         * Therefore the coordinate system has to be rotated so that the z-direction
         * is always specific.
         */
        using namespace cursor::tools;
        cptCurrent1D(twistVectorFieldAxes<PMacc::math::CT::Int < 1, 2, 0 > >(cursorJ), rotateOrigin < 1, 2, 0 > (line), cellSize.x());
        cptCurrent1D(twistVectorFieldAxes<PMacc::math::CT::Int < 2, 0, 1 > >(cursorJ), rotateOrigin < 2, 0, 1 > (line), cellSize.y());
        cptCurrent1D(cursorJ, line, cellSize.z());
    }

    /**
     * deposites current in z-direction
     * \param cursorJ cursor pointing at the current density field of the particle's cell
     * \param line trajectory of the particle from to last to the current time step
     * \param cellEdgeLength length of edge of the cell in z-direction
     */
    template<typename CursorJ >
    DINLINE void cptCurrent1D(CursorJ cursorJ,
                              const Line<float3_X>& line,
                              const float_X cellEdgeLength)
    {
        /* Check if particle position in previous step was greater or
         * smaller than current position.
         *
         * If previous position was greater than current position we change our interval
         * from [begin,end) to [begin+1,end+1).
         */
        const int offset_i = line.m_pos0.x() > line.m_pos1.x() ? 1 : 0;
        const int offset_j = line.m_pos0.y() > line.m_pos1.y() ? 1 : 0;
        const int offset_k = line.m_pos0.z() > line.m_pos1.z() ? 1 : 0;

        /* pick every cell in the xy-plane that is overlapped by particle's
         * form factor and deposit the current for the cells above and beneath
         * that cell and for the cell itself.
         */
        for (int i = begin + offset_i; i < end + offset_i; ++i)
        {
            for (int j = begin + offset_j; j < end + offset_j; ++j)
            {
                /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version from
                 * Esirkepov paper. All coordinates are rotated before thus we can
                 * always use C style W(i,j,k,2).
                 */
                float_X tmp =
                    S0(line, i, 0) * S0(line, j, 1) +
                    float_X(0.5) * DS(line, i, 0) * S0(line, j, 1) +
                    float_X(0.5) * S0(line, i, 0) * DS(line, j, 1) +
                    (float_X(1.0) / float_X(3.0)) * DS(line, i, 0) * DS(line, j, 1);

                float_X accumulated_J = float_X(0.0);
                for (int k = begin + offset_k; k < end + offset_k; ++k)
                {
                    float_X W = DS(line, k, 2) * tmp;
                    /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
                     * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1) */
                    accumulated_J += -this->charge * (float_X(1.0) / float_X(CELL_VOLUME * DELTA_T)) * W * cellEdgeLength;
                    /* the branch divergence here still over-compensates for the fewer collisions in the (expensive) atomic adds */
                    if (accumulated_J != float_X(0.0))
                        atomicAddWrapper(&((*cursorJ(i, j, k)).z()), accumulated_J);
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
};

} //namespace currentSolver

} //namespace picongpu

#include "fields/currentDeposition/Esirkepov/Esirkepov2D.hpp"
