/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Wen Fu,
 *                     Marco Garten, Alexander Grund, Richard Pausch,
 *                     Lennert Sprenger
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

#include <pmacc/math/operation.hpp>


namespace picongpu
{
    namespace particles
    {
        /** move the particle to the new position
         *
         * in case the new postion is outside of the cell we adjust
         * the cell index and the multiMask of the particle.
         *
         * @tparam T_Particle particle type
         * @param particle particle handle
         * @param newPos new position relative to the current cell, in units of cells
         * newPos must be [-1, 2)
         * @return whether the particle has left the orignal cell
         */
        template<typename T_Particle>
        HDINLINE bool moveParticle(T_Particle& particle, floatD_X newPos)
        {
            using TVec = MappingDesc::SuperCellSize;


            floatD_X pos = newPos;
            const int particleCellIdx = particle[localCellIdx_];

            DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec>(particleCellIdx));

            DataSpace<simDim> dir;
            for(uint32_t i = 0; i < simDim; ++i)
            {
                /* ATTENTION we must handle float rounding errors
                 * pos in range [-1;2)
                 *
                 * If pos is negative and very near to 0 (e.g. pos < -1e-8)
                 * and we move pos with pos+=1.0 back to normal in cell postion
                 * we get a rounding error and pos is assigned to 1. This breaks
                 * our in cell definition range [0,1)
                 *
                 * if pos negativ moveDir is set to -1
                 * if pos positive and >1 moveDir is set to +1
                 * 0 (zero) if particle stays in cell
                 */
                float_X moveDir = math::floor(pos[i]);
                /* shift pos back to cell range [0;1)*/
                pos[i] -= moveDir;
                /* check for rounding errors and correct them
                 * if position now is 1 we have a rounding error
                 *
                 * We correct moveDir that we not have left the cell
                 */
                const float_X valueCorrector = math::floor(pos[i]);
                /* One has also to correct moveDir for the following reason:
                 * Imagine a new particle moves to -1e-20, leaving the cell to the left,
                 * setting moveDir to -1.
                 * The new in-cell position will be -1e-20 + 1.0,
                 * which can flip to 1.0 (wrong value).
                 * We move the particle back to the old cell at position 0.0 and
                 * moveDir has to be corrected back, too (add +1 again).*/
                moveDir += valueCorrector;
                /* If we have corrected moveDir we must set pos to 0 */
                pos[i] -= valueCorrector;
                dir[i] = precisionCast<int>(moveDir);
            }
            particle[position_] = pos;

            /* new local cell position after particle move
             * can be out of supercell
             */
            localCell += dir;

            /* ATTENTION ATTENTION we cast to unsigned, this means that a negative
             * direction is know a very very big number, than we compare with supercell!
             *
             * if particle is inside of the supercell the **unsigned** representation
             * of dir is always >= size of the supercell
             */
            for(uint32_t i = 0; i < simDim; ++i)
                dir[i] *= precisionCast<uint32_t>(localCell[i]) >= precisionCast<uint32_t>(TVec::toRT()[i]) ? 1 : 0;

            /* if partice is outside of the supercell we use mod to
            * set particle at cell supercellSize to 1
            * and partticle at cell -1 to supercellSize-1
            * % (mod) can't use with negativ numbers, we add one supercellSize to hide this
            *
            localCell.x() = (localCell.x() + TVec::x) % TVec::x;
            localCell.y() = (localCell.y() + TVec::y) % TVec::y;
            localCell.z() = (localCell.z() + TVec::z) % TVec::z;
            */

            /*dir is only +1 or -1 if particle is outside of supercell
             * y=cell-(dir*superCell_size)
             * y=0 if dir==-1
             * y=superCell_size if dir==+1
             * for dir 0 localCel is not changed
             */
            localCell -= (dir * TVec::toRT());
            /*calculate one dimensional cell index*/
            particle[localCellIdx_] = DataSpaceOperations<TVec::dim>::template map<TVec>(localCell);

            /* [ dir + int(dir < 0)*3 ] == [ (dir + 3) %3 = y ]
             * but without modulo
             * y=0 for dir = 0
             * y=1 for dir = 1
             * y=2 for dir = -1
             */
            int direction = 1;
            uint32_t exchangeType = 1; // see inlcude/pmacc/types.h for RIGHT, BOTTOM and BACK
            for(uint32_t i = 0; i < simDim; ++i)
            {
                direction += (dir[i] == -1 ? 2 : dir[i]) * exchangeType;
                exchangeType *= 3; // =3^i (1=RIGHT, 3=BOTTOM; 9=BACK)
            }

            particle[multiMask_] = direction;

            return direction >= 2;
        }

    } // namespace particles
} // namespace picongpu
