/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Wen Fu,
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
         * in case the new position is outside of the cell we adjust
         * the cell index and the multiMask of the particle.
         *
         * @tparam T_Particle particle type
         * @param particle particle handle
         * @param newPos new position relative to the current cell, in units of cells
         * newPos must be [-1.0, 2.0)
         * @return whether the particle has left the original cell
         */
        template<typename T_Particle>
        HDINLINE bool moveParticle(T_Particle& particle, floatD_X newPos)
        {
            using TVec = MappingDesc::SuperCellSize;


            /* ATTENTION we must handle float rounding errors
             * pos in range [-1.0;2.0)
             *
             * If pos is negative and very near to 0 (e.g. pos < -1e-8)
             * and we move pos with pos+=1.0 back to normal in cell position
             * we get a rounding error and pos is assigned to 1. This breaks
             * our in cell definition range [0,1)
             *
             * We remove 0.5 from position and later add back after we adjusted the new position depending on the
             * direction the particle is leaving the cell.
             * The floating point precision is equal for -0.5 and 0.5.
             */
#if(BOOST_COMP_HIP)
            // workaround for a broken HIP optimization
            // https://github.com/ComputationalRadiationPhysics/picongpu/issues/4561
            volatile
#else
            constexpr
#endif
                auto shift
                = 0.5_X;
            floatD_X pos = newPos - shift;

            DataSpace<simDim> dir;

            for(uint32_t i = 0; i < simDim; ++i)
            {
                // pos in range [-1.5;1.5)
                float_X moveDir = 0.0_X;

                // compiler will use here the ternary operator if necessary
                if(pos[i] < -0.5_X)
                    moveDir = -1.0_X;
                if(pos[i] >= 0.5_X)
                    moveDir = 1.0_X;

                pos[i] -= moveDir;
                particle[position_][i] = pos[i] + shift;
                dir[i] = precisionCast<int>(moveDir);
            }

            // direction is used for multimask where 1 mean particle is valid
            int newMultimask = 1;

            /* multimask and localCell index must only be updated if the particle is moving out of the cell.
             * This is reducing the global memory pressure by performing only necessary memory writes.
             */
            if(dir != DataSpace<simDim>::create(0))
            {
                const int particleCellIdx = particle[localCellIdx_];

                DataSpace<TVec::dim> localCell(DataSpaceOperations<TVec::dim>::template map<TVec>(particleCellIdx));
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
                    dir[i] = precisionCast<uint32_t>(localCell[i]) >= precisionCast<uint32_t>(TVec::toRT()[i]) ? dir[i]
                                                                                                               : 0;

                /* if partice is outside of the supercell we use mod to
                 * set particle at cell supercellSize to 1
                 * and partticle at cell -1 to supercellSize-1
                 * % (mod) can't use with negativ numbers, we add one supercellSize to hide this
                 *
                 * localCell.x() = (localCell.x() + TVec::x) % TVec::x;
                 * localCell.y() = (localCell.y() + TVec::y) % TVec::y;
                 * localCell.z() = (localCell.z() + TVec::z) % TVec::z;
                 *
                 * dir is only +1 or -1 if particle is outside of supercell
                 * localCell = localCell - (dir*superCell_size)
                 * localCell = 0 if dir==-1
                 * localCell = superCell_size - 1 if dir==+1
                 * for dir 0 localCel is not changed
                 */
                localCell -= (dir * TVec::toRT());
                // update one dimensional cell index
                particle[localCellIdx_] = DataSpaceOperations<TVec::dim>::template map<TVec>(localCell);

                // see inlcude/pmacc/type/Exchnages.hpp for RIGHT, BOTTOM and BACK
                uint32_t exchangeType = 1;

                /* transform direction vector into a exchange id
                 *
                 * newMultimask:
                 *   0 == is not possible because each particle processed by this function must be a valid particle
                 *   1 == valid particle whihc is not leaving the supercell
                 *   2 >= valid particle which is leaving the supercell into the direction (newMultimask - 1)
                 */
                for(uint32_t i = 0; i < simDim; ++i)
                {
                    newMultimask += (dir[i] == -1 ? 2 : dir[i]) * exchangeType;
                    exchangeType *= 3; // =3^i (1=RIGHT, 3=BOTTOM; 9=BACK)
                }

                // change multimask only if the particle is leaving the supercell
                if(newMultimask >= 2)
                    particle[multiMask_] = newMultimask;
            }
            return newMultimask >= 2;
        }

    } // namespace particles
} // namespace picongpu
