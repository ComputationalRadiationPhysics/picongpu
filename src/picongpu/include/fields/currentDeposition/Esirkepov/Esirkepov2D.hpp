/**
 * Copyright 2014-2017 Axel Huebl, Heiko Burau, Rene Widera
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
#include "pmacc_types.hpp"
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
    static constexpr int supp = ParticleAssign::support;

    static constexpr int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
    static constexpr int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef typename PMacc::math::CT::make_Int<DIM2, currentLowerMargin>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<DIM2, currentUpperMargin>::type UpperMargin;

    static constexpr int begin = -currentLowerMargin + 1;
    static constexpr int end = begin + supp;

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

        DataSpace<simDim> gridShift;
        /* Define in which direction the particle leaves the cell.
         * It is not important whether the particle move over the positive or negative
         * cell border.
         *
         * 0 == stay in cell
         * 1 == leave cell
         */
        DataSpace<DIM2> leaveCell;

        /* calculate the offset for the virtual coordinate system */
        for(int d=0; d<simDim; ++d)
        {
            int iStart;
            int iEnd;
            constexpr bool isSupportEven = ( supp % 2 == 0 );
            RelayPoint< isSupportEven >()(
                iStart,
                iEnd,
                line.m_pos0[d],
                line.m_pos1[d]
            );
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
            leaveCell,
            cursorJ,
            line,
            cellSize.x()
        );
        cptCurrent1D(DataSpace<DIM2>(
            leaveCell[1],leaveCell[0]),
            twistVectorFieldAxes<PMacc::math::CT::Int < 1, 0 > >(cursorJ),
            rotateOrigin < 1, 0 > (line),
            cellSize.y()
        );
        cptCurrentZ(
            leaveCell,
            cursorJ,
            line,
            velocity.z()
        );
    }

    /**
     * deposites current in z-direction
     * \param leaveCell vector with information (for each direction) if the particle is leaving the cel
     * \param cursorJ cursor pointing at the current density field of the particle's cell
     * \param line trajectory of the particle from to last to the current time step
     * \param cellEdgeLength length of edge of the cell in z-direction
     */
    template<typename CursorJ >
    DINLINE void cptCurrent1D(const DataSpace<simDim>& leaveCell,
                              CursorJ cursorJ,
                              const Line<float2_X>& line,
                              const float_X cellEdgeLength)
    {
        /* skip calculation if the particle is not moving in x direction */
        if(line.m_pos0[0] == line.m_pos1[0])
            return;

        for(int j = begin; j < end + 1; ++j)
            if(j < end + leaveCell[1])
            {
                /* This is the implementation of the FORTRAN W(i,j,k,1)/ C style W(i,j,k,0) version from
                 * Esirkepov paper. All coordinates are rotated before thus we can
                 * always use C style W(i,j,k,0).
                 */
                float_X tmp = S0(line, j, 1) + float_X(0.5) * DS(line, j, 1);

                float_X accumulated_J = float_X(0.0);
                /* attention: inner loop has no upper bound `end + 1` because
                 * the current for the point `end` is always zero,
                 * therefore we skip the calculation
                 */
                for(int i = begin; i < end; ++i)
                    if(i < end + leaveCell[0] - 1)
                    {
                        float_X W = DS(line, i, 0) * tmp;
                        /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
                         * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1) */
                        accumulated_J += -this->charge * (float_X(1.0) / float_X(CELL_VOLUME * DELTA_T)) * W * cellEdgeLength;
                        atomicAddWrapper(&((*cursorJ(i, j)).x()), accumulated_J);
                    }
            }

    }

    template<typename CursorJ >
    DINLINE void cptCurrentZ(const DataSpace<simDim>& leaveCell,
                             CursorJ cursorJ,
                             const Line<float2_X>& line,
                             const float_X v_z)
    {
        if(v_z == float_X( 0.0 ))
                return;

        for(int j = begin; j < end + 1; ++j)
            if(j < end + leaveCell[1])
            {

                for(int i = begin; i < end + 1; ++i)
                    if(i < end + leaveCell[0])
                    {
                        float_X W = S0(line, i, 0) * S0(line, j, 1) +
                            float_X(0.5) * DS(line, i, 0) * S0(line, j, 1) +
                            float_X(0.5) * S0(line, i, 0) * DS(line, j, 1)+
                            (float_X(1.0) / float_X(3.0)) * DS(line, i, 0) * DS(line, j, 1);

                        const float_X j_z = this->charge * (float_X(1.0) / float_X(CELL_VOLUME)) * W * v_z;
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

    static PMacc::traits::StringProperty getStringProperties()
    {
        PMacc::traits::StringProperty propList( "name", "Esirkepov" );
        return propList;
    }
};

} //namespace currentSolver

} //namespace picongpu
