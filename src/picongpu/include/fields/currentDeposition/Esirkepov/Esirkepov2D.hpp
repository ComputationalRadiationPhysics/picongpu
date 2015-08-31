/**
 * Copyright 2014 Axel Huebl, Heiko Burau, Rene Widera
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
#include "fields/currentDeposition/Esirkepov/Esirkepov.hpp"
#include "fields/currentDeposition/Esirkepov/Line.hpp"
#include "algorithms/Velocity.hpp"

namespace picongpu
{
namespace currentSolver
{
using namespace PMacc;

/**
 * \class Esirkepov implements the current deposition algorithm from T.Zh. Esirkepov
 * for an arbitrary particle assign function given as a template parameter.
 * See available shapes at "intermediateLib/particleShape".
 * paper: "Exact charge conservation scheme for Particle-in-Cell simulation
 *  with an arbitrary form-factor"
 */
template<typename T_ParticleShape>
struct Esirkepov<T_ParticleShape, DIM2>
{
    typedef typename T_ParticleShape::ChargeAssignment ParticleAssign;
    static const int supp = ParticleAssign::support;

    static const int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
    static const int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef typename PMacc::math::CT::make_Int<DIM2, currentLowerMargin>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<DIM2, currentUpperMargin>::type UpperMargin;

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

    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos,
                            const VelType velocity,
                            const ChargeType charge, const float_X deltaTime)
    {
        this->charge = charge;
        const float2_X deltaPos = float2_X(velocity.x() * deltaTime / cellSize.x(),
                                           velocity.y() * deltaTime / cellSize.y());
        const PosType oldPos = pos - deltaPos;
        Line<float2_X> line(oldPos, pos);
        BOOST_AUTO(cursorJ, dataBoxJ.toCursor());

        if (supp % 2 == 1)
        {
            /* odd support
             * we need only for odd supports a shift because for even supports
             * we have always the grid range [-support/2+1;support/2] if we have
             * no moving particle
             *
             * With this coordinate shift we only look to the support of one
             * particle which is not moving. Moving praticles coordinate shifts
             * are handled later (named offset_*)
             */

            /* for any direction
             * if pos> 0.5
             * shift curser+1 and new_pos=old_pos-1
             *
             * floor(pos*2.0) is equal (pos > 0.5)
             */
            float2_X coordinate_shift(
                                      float_X(math::floor(pos.x() * float_X(2.0))),
                                      float_X(math::floor(pos.y() * float_X(2.0)))
                                      );
            cursorJ = cursorJ(
                              PMacc::math::Int < 2 > (
                                                      coordinate_shift.x(),
                                                      coordinate_shift.y()
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
        cptCurrent1D(cursorJ, line, cellSize.x());
        cptCurrent1D(twistVectorFieldAxes<PMacc::math::CT::Int < 1, 0 > >(cursorJ), rotateOrigin < 1, 0 > (line), cellSize.y());
        cptCurrentZ(cursorJ, line, velocity.z());
    }

    /**
     * deposites current in z-direction
     * \param cursorJ cursor pointing at the current density field of the particle's cell
     * \param line trajectory of the particle from to last to the current time step
     * \param cellEdgeLength length of edge of the cell in z-direction
     */
    template<typename CursorJ >
    DINLINE void cptCurrent1D(CursorJ cursorJ,
                              const Line<float2_X>& line,
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


        for (int j = begin + offset_j; j < end + offset_j; ++j)
        {
            /* This is the implementation of the FORTRAN W(i,j,k,1)/ C style W(i,j,k,0) version from
             * Esirkepov paper. All coordinates are rotated before thus we can
             * always use C style W(i,j,k,0).
             */
            float_X tmp = S0(line, j, 1) + float_X(0.5) * DS(line, j, 1);

            float_X accumulated_J = float_X(0.0);
            for (int i = begin + offset_i; i < end + offset_i; ++i)
            {
                float_X W = DS(line, i, 0) * tmp;
                /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
                 * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1) */
                accumulated_J += -this->charge * (float_X(1.0) / float_X(CELL_VOLUME * DELTA_T)) * W * cellEdgeLength;
                /* the branch divergence here still over-compensates for the fewer collisions in the (expensive) atomic adds */
                if (accumulated_J != float_X(0.0))
                    atomicAddWrapper(&((*cursorJ(i, j)).x()), accumulated_J);
            }
        }

    }

    template<typename CursorJ >
    DINLINE void cptCurrentZ(CursorJ cursorJ,
                             const Line<float2_X>& line,
                             const float_X v_z)
    {
        /* Check if particle position in previous step was greater or
         * smaller than current position.
         *
         * If previous position was greater than current position we change our interval
         * from [begin,end) to [begin+1,end+1).
         */
        const int offset_i = line.m_pos0.x() > line.m_pos1.x() ? 1 : 0;
        const int offset_j = line.m_pos0.y() > line.m_pos1.y() ? 1 : 0;


        for (int j = begin + offset_j; j < end + offset_j; ++j)
        {
            for (int i = begin + offset_i; i < end + offset_i; ++i)
            {
                float_X W = S0(line, i, 0) * S0(line, j, 1) +
                    float_X(0.5) * DS(line, i, 0) * S0(line, j, 1) +
                    float_X(0.5) * S0(line, i, 0) * DS(line, j, 1)+
                    (float_X(1.0) / float_X(3.0)) * DS(line, i, 0) * DS(line, j, 1);

                const float_X j_z = this->charge * (float_X(1.0) / float_X(CELL_VOLUME)) * W * v_z;
                if (j_z != float_X(0.0))
                    atomicAddWrapper(&((*cursorJ(i, j)).z()), j_z);
            }
        }

    }

    /** calculate S0 (see paper)
     * @param line element with previous and current position of the particle
     * @param gridPoint used grid point to evaluate assignment shape
     * @param d dimension range {0,1} means {x,y}
     *          different to Esirkepov paper, here we use C style
     */
    DINLINE float_X S0(const Line<float2_X>& line, const float_X gridPoint, const uint32_t d)
    {
        return ParticleAssign()(gridPoint - line.m_pos0[d]);
    }

    /** calculate DS (see paper)
     * @param line element with previous and current position of the particle
     * @param gridPoint used grid point to evaluate assignment shape
     * @param d dimension range {0,1} means {x,y}
     *          different to Esirkepov paper, here we use C style
     */
    DINLINE float_X DS(const Line<float2_X>& line, const float_X gridPoint, const uint32_t d)
    {
        return ParticleAssign()(gridPoint - line.m_pos1[d]) - ParticleAssign()(gridPoint - line.m_pos0[d]);
    }
};

} //namespace currentSolver

} //namespace picongpu
