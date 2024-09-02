/* Copyright 2016-2023 Rene Widera, Sergei Bastrakov
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

#include "picongpu/fields/currentDeposition/EmZ/DepositCurrent.hpp"
#include "picongpu/fields/currentDeposition/EmZ/EmZ.def"
#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"
#include "picongpu/fields/currentDeposition/relayPoint.hpp"

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
            using LowerMargin = typename pmacc::math::CT::make_Int<simDim, currentLowerMargin>::type;
            using UpperMargin = typename pmacc::math::CT::make_Int<simDim, currentUpperMargin>::type;

            PMACC_CASSERT_MSG(
                __EmZ_supercell_or_number_of_guard_supercells_is_too_small_for_stencil,
                sizeof(T_ParticleShape*) // defer assert evaluation
                    && pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentLowerMargin
                    && pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentUpperMargin);

            static constexpr int begin = ParticleAssign::begin;
            static constexpr int end = begin + supp;
            static_assert(ParticleAssign::end + 1 == end);


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
            template<typename DataBoxJ, typename T_Worker>
            DINLINE void operator()(
                T_Worker const& worker,
                DataBoxJ dataBoxJ,
                floatD_X const posEnd,
                float3_X const velocity,
                float_X const charge,
                float_X const /* deltaTime */
            )
            {
                floatD_X deltaPos;
                for(uint32_t d = 0; d < simDim; ++d)
                    deltaPos[d] = (velocity[d] * DELTA_T) / sim.pic.getCellSize()[d];

                /*note: all positions are normalized to the grid*/
                const floatD_X posStart(posEnd - deltaPos);

                // Grid shifts for the start and end positions
                DataSpace<simDim> shiftStart, shiftEnd;
                floatD_X relayPointPosition;

                /* calculate the relay point for the trajectory splitting */
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    constexpr bool isSupportEven = (supp % 2 == 0);
                    relayPointPosition[d]
                        = relayPoint<isSupportEven>(shiftStart[d], shiftEnd[d], posStart[d], posEnd[d]);
                }

                Line<floatD_X> line;
                const float_X chargeDensity = charge / sim.pic.getCellSize().productOfComponents();

                /* Esirkepov implementation for the current deposition */
                emz::DepositCurrent<T_Strategy, ParticleAssign, begin, end> deposit;

                /* calculate positions for the second virtual particle, normalized to cell size */
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    line.m_pos0[d] = posStart[d] - shiftStart[d];
                    line.m_pos1[d] = relayPointPosition[d] - shiftStart[d];
                }

                deposit(worker, dataBoxJ.shift(shiftStart), line, chargeDensity);

                /* detect if there is a second virtual particle */
                const bool twoParticlesNeeded = (shiftStart != shiftEnd);
                if(twoParticlesNeeded)
                {
                    /* calculate positions for the second virtual particle */
                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        /* switched start and end point */
                        line.m_pos1[d] = posEnd[d] - shiftEnd[d];
                        line.m_pos0[d] = relayPointPosition[d] - shiftEnd[d];
                    }
                    deposit(worker, dataBoxJ.shift(shiftEnd), line, chargeDensity);
                }

                /* 2d case requires special handling of Jz as explained in #3889.
                 * Pass dataBoxJ as auto&& to defer evaluation for 3d case.
                 */
                if constexpr(simDim == 2)
                {
                    /* For Jz we consider the whole movement on a step.
                     * Note that this movement is not necessarily on support.
                     * A naive implementation would be to extend the bounds in x, y by 1 in both sides, and use
                     * general assignment function. To optimize it, we redefine shiftEnd as component-wise minimum
                     * between old shiftEnd and shiftStart. We calculate everything relative to the new shiftEnd.
                     * Since it is the minimum in both x and y, the same begin value can be used. Thus, the bounds
                     * only have to be extended by 1 in the max side, not both. Still, the general assignment
                     * function has to be used.
                     */
                    for(uint32_t d = 0; d < simDim; ++d)
                    {
                        shiftEnd[d] = math::min(shiftStart[d], shiftEnd[d]);
                        line.m_pos0[d] = posStart[d] - shiftEnd[d];
                        line.m_pos1[d] = posEnd[d] - shiftEnd[d];
                    }
                    /* Have to use DIM2, otherwise 3d case wouldn't compile due to
                     * no computeCurrentZ() method.
                     * In this case it is parsed even though the if condition is false and dataBoxJ.
                     * As we are generally not on support, use T_ParticleShape::ChargeAssignment and not
                     * ParticleAssign.
                     */
                    emz::DepositCurrent<T_Strategy, typename T_ParticleShape::ChargeAssignment, begin, end + 1, DIM2>
                        depositZ;
                    depositZ.computeCurrentZ(worker, dataBoxJ.shift(shiftEnd), line, velocity.z() * chargeDensity);
                }
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "EmZ (Esirkepov-Zigzag, EZ)");
                return propList;
            }
        };

    } // namespace currentSolver

} // namespace picongpu
