/* Copyright 2014-2022 Axel Huebl, Heiko Burau, Rene Widera
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

#include "picongpu/algorithms/Velocity.hpp"
#include "picongpu/fields/currentDeposition/Esirkepov/Esirkepov.hpp"
#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"
#include "picongpu/fields/currentDeposition/Esirkepov/TrajectoryAssignmentShapeFunction.hpp"
#include "picongpu/fields/currentDeposition/Esirkepov/bitPacking.hpp"

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
        struct Esirkepov<T_ParticleShape, T_Strategy, DIM2>
        {
            using ParticleAssign = typename T_ParticleShape::ChargeAssignment;
            static constexpr int supp = ParticleAssign::support;

            static constexpr int currentLowerMargin = supp / 2 + 1 - (supp + 1) % 2;
            static constexpr int currentUpperMargin = (supp + 1) / 2 + 1;
            using LowerMargin = typename pmacc::math::CT::make_Int<2U, currentLowerMargin>::type;
            using UpperMargin = typename pmacc::math::CT::make_Int<2U, currentUpperMargin>::type;

            PMACC_CASSERT_MSG(
                __Esirkepov2D_supercell_or_number_of_guard_supercells_is_too_small_for_stencil,
                pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentLowerMargin

                    && pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                        >= currentUpperMargin);

            static constexpr int begin = ParticleAssign::begin;
            static constexpr int end = begin + supp;
            static_assert(ParticleAssign::end == end);

            float_X charge;

            template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType, typename T_Worker>
            DINLINE void operator()(
                T_Worker const& worker,
                DataBoxJ dataBoxJ,
                const PosType pos,
                const VelType velocity,
                const ChargeType charge,
                const float_X deltaTime)
            {
                this->charge = charge;
                const float2_X deltaPos
                    = float2_X(velocity.x() * deltaTime / cellSize.x(), velocity.y() * deltaTime / cellSize.y());
                const PosType oldPos = pos - deltaPos;
                Line<float2_X> line(oldPos, pos);

                DataSpace<simDim> gridShift;
                /* Bitmask used to hold information in if the particle is leaving the assignment cell for each
                 * direction.
                 */
                DataSpace<DIM2> status;

                /* calculate the offset for the virtual coordinate system */
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    int iStart;
                    int iEnd;
                    constexpr bool isSupportEven = (supp % 2 == 0);
                    relayPoint<isSupportEven>(iStart, iEnd, line.m_pos0[d], line.m_pos1[d]);
                    gridShift[d] = iStart < iEnd ? iStart : iEnd; // integer min function
                    bitpacking::set(
                        status[d],
                        bitpacking::Status::START_PARTICLE_IN_ASSIGNMENT_CELL,
                        gridShift[d] == iStart);
                    bitpacking::set(
                        status[d],
                        bitpacking::Status::END_PARTICLE_IN_ASSIGNMENT_CELL,
                        gridShift[d] == iEnd);
                    /* particle is leaving the cell */
                    bitpacking::set(status[d], bitpacking::Status::LEAVE_CELL, iStart != iEnd);
                    /* shift the particle position to the virtual coordinate system */
                    line.m_pos0[d] -= gridShift[d];
                    line.m_pos1[d] -= gridShift[d];
                }
                /* shift current field to the virtual coordinate system */
                auto fieldJ = dataBoxJ.shift(gridShift);

                /**
                 * \brief the following three calls separate the 3D current deposition
                 * into three independent 1D calls, each for one direction and current component.
                 * Therefore the coordinate system has to be rotated so that the z-direction
                 * is always specific.
                 */
                cptCurrent1D(
                    worker,
                    status,
                    makePermutatedFieldValueAccess<pmacc::math::CT::Int<0, 1>>(fieldJ),
                    line,
                    cellSize.x());
                cptCurrent1D(
                    worker,
                    DataSpace<DIM2>(status[1], status[0]),
                    makePermutatedFieldValueAccess<pmacc::math::CT::Int<1, 0>>(fieldJ),
                    rotateOrigin<1, 0>(line),
                    cellSize.y());
                cptCurrentZ(worker, status, fieldJ, line, velocity.z());
            }

            /** deposites current in x-direction (rotated PIConGPU coordinate system)
             *
             * @tparam T_DataBox databox type
             * @tparam T_Permutation compile time permutation vector @see PermutatedFieldValueAccess
             *
             * @param parStatus vector with particle status information for each direction
             * @param jField permuted current field shifted to the particle position
             * @param line trajectory of the particle from to last to the current time step
             * @param cellEdgeLength length of edge of the cell in z-direction
             *
             * @{
             */
            template<typename T_DataBox, typename T_Permutation, typename T_Worker>
            DINLINE void cptCurrent1D(
                T_Worker const& worker,
                const DataSpace<simDim>& parStatus,
                PermutatedFieldValueAccess<T_DataBox, T_Permutation> jField,
                const Line<float2_X>& line,
                const float_X cellEdgeLength)
            {
                /* skip calculation if the particle is not moving in x direction */
                if(line.m_pos0[0] == line.m_pos1[0])
                    return;

                auto shapeI = makeTrajectoryAssignmentShapeFunction(
                    typename T_Strategy::template ShapeOuterLoop<ParticleAssign>{
                        line.m_pos0[0],
                        bitpacking::test(parStatus[0], bitpacking::Status::START_PARTICLE_IN_ASSIGNMENT_CELL)},
                    typename T_Strategy::template ShapeOuterLoop<ParticleAssign>{
                        line.m_pos1[0],
                        bitpacking::test(parStatus[0], bitpacking::Status::END_PARTICLE_IN_ASSIGNMENT_CELL)});

                auto shapeJ = makeTrajectoryAssignmentShapeFunction(
                    typename T_Strategy::template ShapeMiddleLoop<ParticleAssign>{
                        line.m_pos0[1],
                        bitpacking::test(parStatus[1], bitpacking::Status::START_PARTICLE_IN_ASSIGNMENT_CELL)},
                    typename T_Strategy::template ShapeMiddleLoop<ParticleAssign>{
                        line.m_pos1[1],
                        bitpacking::test(parStatus[1], bitpacking::Status::END_PARTICLE_IN_ASSIGNMENT_CELL)});

                /* We multiply with `cellEdgeLength` due to the fact that the attribute for the
                 * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1)
                 */
                const float_X currentSurfaceDensity
                    = this->charge * (1.0_X / float_X(CELL_VOLUME * DELTA_T)) * cellEdgeLength;

                for(int j = begin; j < end + 1; ++j)
                    if(j < end + bitpacking::getValue(parStatus[1], bitpacking::Status::LEAVE_CELL))
                    {
                        const float_X s0j = shapeJ.S0(j);
                        const float_X dsj = shapeJ.S0(j) - s0j;

                        float_X tmp = -currentSurfaceDensity * (s0j + 0.5_X * dsj);

                        auto accumulated_J = 0.0_X;
                        /* attention: inner loop has no upper bound `end + 1` because
                         * the current for the point `end` is always zero,
                         * therefore we skip the calculation
                         */
                        for(int i = begin; i < end; ++i)
                            if(i < end + bitpacking::getValue(parStatus[0], bitpacking::Status::LEAVE_CELL) - 1)
                            {
                                /* This is the implementation of the FORTRAN W(i,j,k,1)/ C style W(i,j,k,0) version
                                 * from Esirkepov paper. All coordinates are rotated before thus we can always use C
                                 * style W(i,j,k,0).
                                 */
                                const float_X W = shapeI.DS(i) * tmp;
                                accumulated_J += W;
                                auto const atomicOp = typename T_Strategy::BlockReductionOp{};
                                atomicOp(worker, jField.template get<0>(i, j), accumulated_J);
                            }
                    }
            }

            template<typename T_JField, typename T_Worker>
            DINLINE void cptCurrentZ(
                T_Worker const& worker,
                const DataSpace<simDim>& parStatus,
                T_JField jField,
                const Line<float2_X>& line,
                const float_X v_z)
            {
                if(v_z == 0.0_X)
                    return;

                auto shapeI = makeTrajectoryAssignmentShapeFunction(
                    typename T_Strategy::template ShapeOuterLoop<ParticleAssign>{
                        line.m_pos0[0],
                        bitpacking::test(parStatus[0], bitpacking::Status::START_PARTICLE_IN_ASSIGNMENT_CELL)},
                    typename T_Strategy::template ShapeOuterLoop<ParticleAssign>{
                        line.m_pos1[0],
                        bitpacking::test(parStatus[0], bitpacking::Status::END_PARTICLE_IN_ASSIGNMENT_CELL)});

                auto shapeJ = makeTrajectoryAssignmentShapeFunction(
                    typename T_Strategy::template ShapeMiddleLoop<ParticleAssign>{
                        line.m_pos0[1],
                        bitpacking::test(parStatus[1], bitpacking::Status::START_PARTICLE_IN_ASSIGNMENT_CELL)},
                    typename T_Strategy::template ShapeMiddleLoop<ParticleAssign>{
                        line.m_pos1[1],
                        bitpacking::test(parStatus[1], bitpacking::Status::END_PARTICLE_IN_ASSIGNMENT_CELL)});

                float_X const currentSurfaceDensityZ = this->charge * (1.0_X / float_X(CELL_VOLUME)) * v_z;
                int const leaveCellJ = bitpacking::getValue(parStatus[1], bitpacking::Status::LEAVE_CELL);

                for(int j = begin; j < end + 1; ++j)
                    if(j < end + leaveCellJ)
                    {
                        const float_X s0j = shapeJ.S0(j);
                        const float_X dsj = shapeJ.S1(j) - s0j;

                        int const leaveCellI = bitpacking::getValue(parStatus[0], bitpacking::Status::LEAVE_CELL);
                        for(int i = begin; i < end + 1; ++i)
                            if(i < end + leaveCellI)
                            {
                                const float_X s0i = shapeI.S0(i);
                                const float_X dsi = shapeI.S1(i) - s0i;
                                float_X W = s0i * s0j + 0.5_X * (dsi * s0j + s0i * dsj) + (1.0_X / 3.0_X) * dsi * dsj;

                                const float_X j_z = W * currentSurfaceDensityZ;
                                auto const atomicOp = typename T_Strategy::BlockReductionOp{};
                                atomicOp(worker, jField(DataSpace<DIM2>(i, j)).z(), j_z);
                            }
                    }
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "Esirkepov");
                return propList;
            }
        };

    } // namespace currentSolver

} // namespace picongpu
