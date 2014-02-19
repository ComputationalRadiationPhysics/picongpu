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
#include "math/vector/UInt.hpp"
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
namespace currentSolverEsirkepov
{
using namespace PMacc;

/**
 * \class Esirkepov implements the current deposition algorithm from T.Zh. Esirkepov
 * for an arbitrary particle assign function given as a template parameter.
 * See available shapes at "intermediateLib/particleShape".
 * paper: "Exact charge conservation scheme for Particle-in-Cell simulation
 *  with an arbitrary form-factor"
 */
template<typename ParticleAssign, typename NumericalCellType>
struct Esirkepov<DIM2,ParticleAssign,NumericalCellType>
{
    static const int supp = ParticleAssign::support;

    static const int currentLowerMargin = supp / 2 + 1;
    static const int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef typename PMacc::math::CT::make_Int<DIM2,currentLowerMargin>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<DIM2,currentUpperMargin>::type UpperMargin;

    /* begin and end border is calculated for a particle with a support which travels
     * to the negative direction.
     * Later on all coordinates shifted thus we can solve the charge calculation
     * independend from the position of the particle. That means we must not look
     * if a particle position is >0.5 oder not (this is done by coordinate shifting to this defined range)
     * 
     * (supp + 1) % 2 is 1 for even supports else 0
     */
    static const int begin = -supp / 2 + (supp + 1) % 2 - 1;
    static const int end = supp / 2;

    float_X charge;

    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos,
                            const VelType velocity,
                            const ChargeType charge, const float2_X& cellSize, const float_X deltaTime)
    {
        this->charge = charge;
        const float2_X deltaPos = float2_X(velocity.x() * deltaTime / cellSize.x(),
                                           velocity.y() * deltaTime / cellSize.y() );
        const PosType oldPos = pos - deltaPos;
        Line<float2_X> line(oldPos, pos);
        BOOST_AUTO(cursorJ, dataBoxJ.toCursor());

        if (speciesParticleShape::ParticleShape::support % 2 == 1)
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
            line.pos0 -= (coordinate_shift);
            line.pos1 -= (coordinate_shift);
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
        /* We need no shifts of the coordinate system because W is defined on point (0,0,0)
         *
         * \todo: W is only on point (0,0,0) if we use Yee, please fix me for CenteredCell
         */


        /* Now we check if particle travel to negativ or positiv direction.
         * If particle travels to positiv direction we change the gridintervall
         * from [begin;end] to [begin+1;end+1].
         * For particle which travel to negativ direction we change nothing, 
         * because begin and end by default are defined for negativ travel particles
         * 
         * We calculate pos0-pos1 because we have done our coordinate shifts with
         * pos1 and need the information if pos0 is left or right of pos1.
         */

        const int offset_y = math::floor(line.pos0.y() - line.pos1.y() + float_X(1.0));

        /* pick every cell in the xy-plane that is overlapped by particle's
         * form factor and deposite the current for the cells above and beneath
         * that cell and for the cell itself.
         */

        for (int y = begin + offset_y; y <= end + offset_y; y++)
        {
            cptCurrentInLineOfCells(cursorJ(0, y), line - float2_X(0., y), cellEdgeLength);
        }
    }

    template<typename CursorJ >
    DINLINE void cptCurrentZ(CursorJ cursorJ,
                             const Line<float2_X>& line,
                             const float_X v_z)
    {
        const int offset_x = math::floor(line.pos0.x() - line.pos1.x() + float_X(1.0));
        const int offset_y = math::floor(line.pos0.y() - line.pos1.y() + float_X(1.0));

        /* pick every cell in the xy-plane that is overlapped by particle's
         * form factor and deposite the current for the cells above and beneath
         * that cell and for the cell itself.
         */
        for (int x = begin + offset_x; x <= end + offset_x; x++)
        {
            for (int y = begin + offset_y; y <= end + offset_y; y++)
            {
                cptCurrentZInCell(cursorJ(x, y), line - float2_X(x, y), v_z);
            }
        }
    }

    /**
     * deposites current in a line of cells in z-direction
     * \param cursorJ cursor pointing at the current density field of the particle's cell
     * \param line trajectory of the particle from to last to the current time step
     * \param cellEdgeLength length of edge of the cell in z-direction
     */
    template<typename CursorJ >
    DINLINE void cptCurrentInLineOfCells(
                                         const CursorJ& cursorJ,
                                         const Line<float2_X>& line,
                                         const float_X cellEdgeLength)
    {

        const float_X tmp = (S0(line.pos0.y()) + float_X(0.5) * DS(line.pos0.y(), line.pos1.y()));


        /* integrate W, which is the divergence of the current density, in x-direction
         * to get the current density j
         */
        const int offset_x = math::floor(line.pos0.x() - line.pos1.x() + float_X(1.0));

        float_X accumulated_J = float_X(0.0);
        for (int i = begin + offset_x; i <= end + offset_x; i++)
        {
            float_X W = DS(line.pos0.x()-i, line.pos1.x()-i) * tmp;
            accumulated_J += -this->charge * (float_X(1.0) / float_X(CELL_VOLUME * DELTA_T)) * W * cellEdgeLength;
            if(accumulated_J!=float_X(0.0))
                atomicAddWrapper(&((*cursorJ(i, 0)).x()), accumulated_J);
        }
    }

    template<typename CursorJ >
    DINLINE void cptCurrentZInCell(
                                   const CursorJ& cursorJ,
                                   const Line<float2_X>& line,
                                   const float_X v_z)
    {


        float_X W = S0(line.pos0.x()) * S0(line.pos0.y()) +
            float_X(0.5) * DS(line.pos0.x(), line.pos1.x()) * S0(line.pos0.y()) +
            float_X(0.5) * S0(line.pos0.x()) * DS(line.pos0.y(), line.pos1.y()) +
            (float_X(1.0) / float_X(3.0)) * DS(line.pos0.x(), line.pos1.x()) * DS(line.pos0.y(), line.pos1.y());

        const float_X j_z = this->charge * (float_X(1.0) / float_X(CELL_VOLUME)) * W * v_z;
        if(j_z!=float_X(0.0))
            atomicAddWrapper(&((*cursorJ(0, 0)).z()), j_z);

    }

    DINLINE float_X S0(const float_X k)
    {
        return ParticleAssign()(k);
    }

    DINLINE float_X DS(const float_X k0, const float_X k1)
    {
        return ParticleAssign()(k1) - ParticleAssign()(k0);
    }
};

} //namespace currentSolverEsirkepov

} //namespace picongpu
