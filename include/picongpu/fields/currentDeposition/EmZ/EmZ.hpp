/* Copyright 2016-2021 Rene Widera
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

#include <pmacc/cuSTL/cursor/Cursor.hpp>

#include "picongpu/fields/currentDeposition/EmZ/EmZ.def"
#include "picongpu/fields/currentDeposition/RelayPoint.hpp"
#include "picongpu/fields/currentDeposition/EmZ/DepositCurrent.hpp"
#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"


namespace picongpu
{
    namespace currentSolver
    {
        template<typename T_ParticleShape, typename T_Strategy>
        struct EmZ
        {
            using ParticleAssign = typename T_ParticleShape::ChargeAssignmentOnSupport;
            static constexpr int supp = ParticleAssign::support;

            static constexpr int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
            static constexpr int currentUpperMargin = (supp + 1) / 2 + 1;
            typedef typename pmacc::math::CT::make_Int<simDim, currentLowerMargin>::type LowerMargin;
            typedef typename pmacc::math::CT::make_Int<simDim, currentUpperMargin>::type UpperMargin;

            PMACC_CASSERT_MSG(
                __EmZ_supercell_or_number_of_guard_supercells_is_too_small_for_stencil,
                pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentLowerMargin
                    && pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentUpperMargin);


            static constexpr int begin = -currentLowerMargin + 1;
            static constexpr int end = begin + supp;


            /** deposit the current of a particle
             *
             * @tparam DataBoxJ any pmacc DataBox
             *
             * @param dataBoxJ box shifted to the cell of particle
             * @param posEnd position of the particle after it is pushed
             * @param velocity velocity of the particle
             * @param charge charge of the particle
             * @param deltaTime time of one time step
             */
            template<typename DataBoxJ, typename T_Acc>
            DINLINE void operator()(
                T_Acc const& acc,
                DataBoxJ dataBoxJ,
                floatD_X const posEnd,
                float3_X const velocity,
                float_X const charge,
                float_X const /* deltaTime */
            )
            {
                floatD_X deltaPos;
                for(uint32_t d = 0; d < simDim; ++d)
                    deltaPos[d] = (velocity[d] * DELTA_T) / cellSize[d];

                /*note: all positions are normalized to the grid*/
                const floatD_X posStart(posEnd - deltaPos);

                DataSpace<simDim> I[2];
                floatD_X relayPoint;

                /* calculate the relay point for the trajectory splitting */
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    constexpr bool isSupportEven = (supp % 2 == 0);
                    relayPoint[d] = RelayPoint<isSupportEven>()(I[0][d], I[1][d], posStart[d], posEnd[d]);
                }

                Line<floatD_X> line;
                const float_X chargeDensity = charge / CELL_VOLUME;

                /* Esirkepov implementation for the current deposition */
                emz::DepositCurrent<typename T_Strategy::BlockReductionOp, ParticleAssign, begin, end> deposit;

                /* calculate positions for the second virtual particle */
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    line.m_pos0[d] = calc_InCellPos(posStart[d], I[0][d]);
                    line.m_pos1[d] = calc_InCellPos(relayPoint[d], I[0][d]);
                }

                const bool twoParticlesNeeded = I[0] != I[1];

                deposit(
                    acc,
                    dataBoxJ.shift(I[0]).toCursor(),
                    line,
                    chargeDensity,
                    velocity.z() * (twoParticlesNeeded ? float_X(0.5) : float_X(1.0)));

                /* detect if there is a second virtual particle */
                if(twoParticlesNeeded)
                {
                    /* calculate positions for the second virtual particle */
                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        /* switched start and end point */
                        line.m_pos1[d] = calc_InCellPos(posEnd[d], I[1][d]);
                        line.m_pos0[d] = calc_InCellPos(relayPoint[d], I[1][d]);
                    }
                    deposit(acc, dataBoxJ.shift(I[1]).toCursor(), line, chargeDensity, velocity.z() * float_X(0.5));
                }
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "EmZ");
                return propList;
            }


            /** get normalized in cell particle position
             *
             * @param x position of the particle
             * @param i shift of grid (only integral positions are allowed)
             * @return in cell position
             */
            DINLINE float_X calc_InCellPos(const float_X x, const float_X i) const
            {
                return x - i;
            }
        };

    } // namespace currentSolver

} // namespace picongpu
