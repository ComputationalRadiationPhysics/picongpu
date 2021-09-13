/* Copyright 2021 Sergei Bastrakov, Lennert Sprenger
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

#include "picongpu/particles/boundary/ApplyImpl.hpp"
#include "picongpu/particles/boundary/Kind.hpp"
#include "picongpu/particles/boundary/Parameters.hpp"
#include "picongpu/particles/boundary/Utility.hpp"
#include "picongpu/particles/functor/misc/Parametrized.hpp"
#include "picongpu/particles/manipulators/unary/FreeTotalCellOffset.hpp"

#include <cstdint>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            struct ReflectParticleIfOutside : public functor::misc::Parametrized<Parameters>
            {
                //! some name is required
                static constexpr char const* name = "reflectParticleIfOutside";

                /** Process the current particle located in the given cell
                 *
                 * @param offsetToTotalOrigin offset of particle cell in the total domain
                 * @param particle handle of particle to process (can be used to change attribute values)
                 */
                template<typename T_Particle>
                HDINLINE void operator()(DataSpace<simDim> const& offsetToTotalOrigin, T_Particle& particle)
                {
                    
                    for(uint32_t d = 0; d < simDim; d++)
                        if((offsetToTotalOrigin[d] < m_parameters.beginInternalCellsTotal[d])
                            || (offsetToTotalOrigin[d] >= m_parameters.endInternalCellsTotal[d]))
                            {  
                                floatD_X pos = particle[position_];
                                printf("old %f %f %f\n", pos.x(), pos.y(), pos.z());
                                if (offsetToTotalOrigin[d] >= m_parameters.endInternalCellsTotal[d])
                                    pos[d] = -pos[d];
                                if (offsetToTotalOrigin[d] < m_parameters.beginInternalCellsTotal[d])
                                    pos[d] = 2-pos[d];
                                printf("new %f %f %f\n", pos.x(), pos.y(), pos.z());
                                floatD_X newPos = pos;
                                particle[momentum_][d] = -particle[momentum_][d];   
                                multi(particle, newPos);
                            }
                        

                }

                template<typename T_Particle>
                HDINLINE void multi(T_Particle& particle, floatD_X newPos)
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

                    printf("endof multi\n");
                                        
                    /*
                    if(direction >= 2)
                    {
                        kernel::atomicAllExch(acc, &mustShift, 1, ::alpaka::hierarchy::Threads{});
                    }*/
                }
            };

            //! Functor to apply reflecting boundary condition to particle species
            template<>
            struct ApplyImpl<Kind::Reflecting>
            {
                /** Apply reflecting boundary conditions along the given outer boundary
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param species particle species
                 * @param exchangeType exchange describing the active boundary
                 * @param currentStep current time iteration
                 */
                template<typename T_Species>
                void operator()(T_Species& species, uint32_t exchangeType, uint32_t currentStep)
                {
                    /* The rest of this function is not optimal performance-wise.
                     * However it is only used when a user set a positive offset, so tolerable.
                     * It processes all particles in manipulate and fillAllGaps() instead of working on the active area
                     * specifically. Currently it would also go over several times if multiple boundaries are
                     * absorbing.
                     */
                    pmacc::DataSpace<simDim> beginInternalCellsTotal, endInternalCellsTotal;
                    getInternalCellsTotal(species, exchangeType, &beginInternalCellsTotal, &endInternalCellsTotal);
                    ReflectParticleIfOutside::parameters().beginInternalCellsTotal = beginInternalCellsTotal;
                    ReflectParticleIfOutside::parameters().endInternalCellsTotal = endInternalCellsTotal;
                    auto const mapperFactory = getMapperFactory(species, exchangeType);
                    using Manipulator = manipulators::unary::FreeTotalCellOffset<ReflectParticleIfOutside>;
                    particles::manipulate<Manipulator, T_Species>(currentStep, mapperFactory);
                    //particles::manipulate<Manipulator, T_Species, particles::filter::All, CORE + BORDER + GUARD>(currentStep); //, mapperFactory);
                    
                    species.template shiftBetweenSupercells<CORE + BORDER + GUARD /* GUARD */>();
                    
                }
            };


        } // namespace boundary
    } // namespace particles
} // namespace picongpu
