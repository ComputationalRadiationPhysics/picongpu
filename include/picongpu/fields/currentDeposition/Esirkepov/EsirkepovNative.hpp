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
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/cuSTL/cursor/Cursor.hpp>
#include <pmacc/cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include <pmacc/cuSTL/cursor/compile-time/SafeCursor.hpp>

#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"


namespace picongpu
{
    namespace currentSolver
    {
        /**
         * Implements the current deposition algorithm from T.Zh. Esirkepov
         *
         * for an arbitrary particle assign function given as a template parameter.
         * See available shapes at "intermediateLib/particleShape".
         * paper: "Exact charge conservation scheme for Particle-in-Cell simulation
         *  with an arbitrary form-factor"
         */
        template<typename T_ParticleShape, typename T_Strategy>
        struct EsirkepovNative
        {
            using ParticleAssign = typename T_ParticleShape::ChargeAssignment;
            static constexpr int supp = ParticleAssign::support;

            static constexpr int currentLowerMargin = supp / 2 + 1;
            static constexpr int currentUpperMargin = (supp + 1) / 2 + 1;
            typedef pmacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
            typedef pmacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

            PMACC_CASSERT_MSG(
                __EsirkepovNative_supercell_or_number_of_guard_supercells_is_too_small_for_stencil,
                pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentLowerMargin
                    && pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentUpperMargin);

            /* iterate over all grid points */
            static constexpr int begin = -currentLowerMargin;
            static constexpr int end = currentUpperMargin + 1;

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
                const float3_X deltaPos = velocity * deltaTime / cellSize;
                const PosType oldPos = pos - deltaPos;
                const Line<float3_X> line(oldPos, pos);
                auto cursorJ = dataBoxJ.toCursor();

                /**
                 * \brief the following three calls separate the 3D current deposition
                 * into three independent 1D calls, each for one direction and current component.
                 * Therefore the coordinate system has to be rotated so that the z-direction
                 * is always specific.
                 */

                using namespace cursor::tools;
                cptCurrent1D(
                    acc,
                    twistVectorFieldAxes<pmacc::math::CT::Int<1, 2, 0>>(cursorJ),
                    rotateOrigin<1, 2, 0>(line),
                    cellSize.x());
                cptCurrent1D(
                    acc,
                    twistVectorFieldAxes<pmacc::math::CT::Int<2, 0, 1>>(cursorJ),
                    rotateOrigin<2, 0, 1>(line),
                    cellSize.y());
                cptCurrent1D(acc, cursorJ, line, cellSize.z());
            }

            /**
             * deposites current in z-direction
             * \param cursorJ cursor pointing at the current density field of the particle's cell
             * \param line trajectory of the particle from to last to the current time step
             * \param cellEdgeLength length of edge of the cell in z-direction
             */
            template<typename CursorJ, typename T_Acc>
            DINLINE void cptCurrent1D(
                T_Acc const& acc,
                CursorJ cursorJ,
                const Line<float3_X>& line,
                const float_X cellEdgeLength)
            {
                /* pick every cell in the xy-plane that is overlapped by particle's
                 * form factor and deposit the current for the cells above and beneath
                 * that cell and for the cell itself.
                 */
                for(int i = begin; i < end; ++i)
                {
                    for(int j = begin; j < end; ++j)
                    {
                        float_X tmp = S0(line, i, 1) * S0(line, j, 2) + float_X(0.5) * DS(line, i, 1) * S0(line, j, 2)
                            + float_X(0.5) * S0(line, i, 1) * DS(line, j, 2)
                            + (float_X(1.0) / float_X(3.0)) * DS(line, i, 1) * DS(line, j, 2);

                        float_X accumulated_J = float_X(0.0);
                        for(int k = begin; k < end; ++k)
                        {
                            const float_X W = DS(line, k, 3) * tmp;
                            /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
                             * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1) */
                            accumulated_J += -this->charge * (float_X(1.0) / float_X(CELL_VOLUME * DELTA_T)) * W
                                * cellEdgeLength;
                            auto const atomicOp = typename T_Strategy::BlockReductionOp{};
                            atomicOp(acc, (*cursorJ(i, j, k)).z(), accumulated_J);
                        }
                    }
                }
            }

            /** calculate S0 (see paper)
             * @param line element with previous and current position of the particle
             * @param gridPoint used grid point to evaluate assignment shape
             * @param d dimension range {1,2,3} means {x,y,z}
             *        same like in Esirkepov paper (FORTAN style)
             */
            DINLINE float_X S0(const Line<float3_X>& line, const float_X gridPoint, const float_X d)
            {
                return ParticleAssign()(gridPoint - line.m_pos0[d - 1]);
            }

            /** calculate DS (see paper)
             * @param line element with previous and current position of the particle
             * @param gridPoint used grid point to evaluate assignment shape
             * @param d dimension range {1,2,3} means {x,y,z}
             *        same like in Esirkepov paper (FORTAN style)
             */
            DINLINE float_X DS(const Line<float3_X>& line, const float_X gridPoint, const float_X d)
            {
                return ParticleAssign()(gridPoint - line.m_pos1[d - 1])
                    - ParticleAssign()(gridPoint - line.m_pos0[d - 1]);
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "Esirkepov");
                propList["param"] = "native implementation";
                return propList;
            }
        };

    } // namespace currentSolver

    namespace traits
    {
        /*Get margin of a solver
         * class must define a LowerMargin and UpperMargin
         */
        template<typename T_ParticleShape, typename T_Strategy>
        struct GetMargin<picongpu::currentSolver::EsirkepovNative<T_ParticleShape, T_Strategy>>
        {
        private:
            using Solver = picongpu::currentSolver::EsirkepovNative<T_ParticleShape, T_Strategy>;

        public:
            using LowerMargin = typename Solver::LowerMargin;
            using UpperMargin = typename Solver::UpperMargin;
        };

    } // namespace traits
} // namespace picongpu
