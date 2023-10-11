/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
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

#include "picongpu/fields/currentDeposition/EmZ/EmZ.def"
#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"
#include "picongpu/fields/currentDeposition/Esirkepov/TrajectoryAssignmentShapeFunction.hpp"
#include "picongpu/fields/currentDeposition/PermutatedFieldValueAccess.hpp"


namespace picongpu
{
    namespace currentSolver
    {
        namespace emz
        {
            template<typename T_Strategy, typename T_ParticleAssign, int T_begin, int T_end>
            struct DepositCurrent<T_Strategy, T_ParticleAssign, T_begin, T_end, DIM3>
            {
                template<typename T_DataBox, typename T_Worker>
                DINLINE void operator()(
                    T_Worker const& worker,
                    T_DataBox fieldJ,
                    const Line<float3_X>& line,
                    const float_X chargeDensity) const
                {
                    /**
                     * \brief the following three calls separate the 3D current deposition
                     * into three independent 1D calls, each for one direction and current component.
                     * Therefore the coordinate system has to be rotated so that the z-direction
                     * is always specific.
                     */
                    cptCurrent1D(
                        worker,
                        makePermutatedFieldValueAccess<pmacc::math::CT::Int<1, 2, 0>>(fieldJ),
                        rotateOrigin<1, 2, 0>(line),
                        cellSize.x() * chargeDensity / DELTA_T);
                    cptCurrent1D(
                        worker,
                        makePermutatedFieldValueAccess<pmacc::math::CT::Int<2, 0, 1>>(fieldJ),
                        rotateOrigin<2, 0, 1>(line),
                        cellSize.y() * chargeDensity / DELTA_T);
                    cptCurrent1D(
                        worker,
                        makePermutatedFieldValueAccess<pmacc::math::CT::Int<0, 1, 2>>(fieldJ),
                        line,
                        cellSize.z() * chargeDensity / DELTA_T);
                }

                /** deposites current in z-direction
                 *
                 * @tparam T_DataBox databox type
                 * @tparam T_Permutation compile time permutation vector @see PermutatedFieldValueAccess
                 *
                 * @param jField permuted current field shifted to the particle position
                 * @param line trajectory of the virtual particle
                 * @param currentSurfaceDensity surface density
                 */
                template<typename T_DataBox, typename T_Permutation, typename T_Line, typename T_Worker>
                DINLINE void cptCurrent1D(
                    T_Worker const& worker,
                    PermutatedFieldValueAccess<T_DataBox, T_Permutation> jField,
                    const T_Line& line,
                    const float_X currentSurfaceDensity) const
                {
                    if(line.m_pos0[2] == line.m_pos1[2])
                        return;

                    auto const shapeI = makeTrajectoryAssignmentShapeFunction(
                        typename T_Strategy::template ShapeOuterLoop<T_ParticleAssign>{line.m_pos0[0], true},
                        typename T_Strategy::template ShapeOuterLoop<T_ParticleAssign>{line.m_pos1[0], true});

                    auto const shapeJ = makeTrajectoryAssignmentShapeFunction(
                        typename T_Strategy::template ShapeMiddleLoop<T_ParticleAssign>{line.m_pos0[1], true},
                        typename T_Strategy::template ShapeMiddleLoop<T_ParticleAssign>{line.m_pos1[1], true});

                    auto const shapeK = makeTrajectoryAssignmentShapeFunction(
                        typename T_Strategy::template ShapeInnerLoop<T_ParticleAssign>{line.m_pos0[2], true},
                        typename T_Strategy::template ShapeInnerLoop<T_ParticleAssign>{line.m_pos1[2], true});

                    /* pick every cell in the xy-plane that is overlapped by particle's
                     * form factor and deposit the current for the cells above and beneath
                     * that cell and for the cell itself.
                     */
                    for(int i = T_begin; i < T_end; ++i)
                    {
                        const float_X s0i = shapeI.S0(i);
                        const float_X dsi = shapeI.S1(i) - s0i;
                        for(int j = T_begin; j < T_end; ++j)
                        {
                            const float_X s0j = shapeJ.S0(j);
                            const float_X dsj = shapeJ.S1(j) - s0j;

                            float_X tmp = -currentSurfaceDensity
                                * (s0i * s0j + 0.5_X * (dsi * s0j + s0i * dsj) + (1.0_X / 3.0_X) * dsj * dsi);

                            auto accumulated_J = 0.0_X;
                            for(int k = T_begin; k < T_end - 1; ++k)
                            {
                                /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version
                                 * from Esirkepov paper. All coordinates are rotated before thus we can always use C
                                 * style W(i,j,k,2).
                                 */
                                const float_X W = shapeK.DS(k) * tmp;
                                accumulated_J += W;
                                auto const atomicOp = typename T_Strategy::BlockReductionOp{};
                                atomicOp(worker, jField.template get<2>(i, j, k), accumulated_J);
                            }
                        }
                    }
                }
            };

            template<typename T_Strategy, typename T_ParticleAssign, int T_begin, int T_end>
            struct DepositCurrent<T_Strategy, T_ParticleAssign, T_begin, T_end, DIM2>
            {
                /** Deposit Jx and Jy.
                 *
                 * In 2d, we have to handle Jz differently from Jx, Jy.
                 * It is done in computeCurrentZ() which has to be explicitly called by a user.
                 * This it different from 3d, where only calling operator() is needed.
                 */
                template<typename T_DataBox, typename T_Worker>
                DINLINE void operator()(
                    T_Worker const& worker,
                    T_DataBox fieldJ,
                    const Line<float2_X>& line,
                    const float_X chargeDensity) const
                {
                    cptCurrent1D(
                        worker,
                        makePermutatedFieldValueAccess<pmacc::math::CT::Int<0, 1>>(fieldJ),
                        line,
                        cellSize.x() * chargeDensity / DELTA_T);
                    cptCurrent1D(
                        worker,
                        makePermutatedFieldValueAccess<pmacc::math::CT::Int<1, 0>>(fieldJ),
                        rotateOrigin<1, 0>(line),
                        cellSize.y() * chargeDensity / DELTA_T);
                }

                /** deposites current in x-direction
                 *
                 * @param jField current density field of the particle's cell
                 * @param line trajectory of the virtual particle
                 * @param currentSurfaceDensity surface density
                 */
                template<typename T_DataBox, typename T_Permutation, typename T_Line, typename T_Worker>
                DINLINE void cptCurrent1D(
                    T_Worker const& worker,
                    PermutatedFieldValueAccess<T_DataBox, T_Permutation> jField,
                    const T_Line& line,
                    const float_X currentSurfaceDensity) const
                {
                    if(line.m_pos0[0] == line.m_pos1[0])
                        return;

                    auto const shapeI = makeTrajectoryAssignmentShapeFunction(
                        typename T_Strategy::template ShapeInnerLoop<T_ParticleAssign>{line.m_pos0[0], true},
                        typename T_Strategy::template ShapeInnerLoop<T_ParticleAssign>{line.m_pos1[0], true});

                    auto const shapeJ = makeTrajectoryAssignmentShapeFunction(
                        typename T_Strategy::template ShapeMiddleLoop<T_ParticleAssign>{line.m_pos0[1], true},
                        typename T_Strategy::template ShapeMiddleLoop<T_ParticleAssign>{line.m_pos1[1], true});

                    for(int j = T_begin; j < T_end; ++j)
                    {
                        const float_X s0j = shapeJ.S0(j);
                        const float_X dsj = shapeJ.S1(j) - s0j;

                        float_X tmp = -currentSurfaceDensity * (s0j + 0.5_X * dsj);

                        auto accumulated_J = 0.0_X;
                        for(int i = T_begin; i < T_end - 1; ++i)
                        {
                            /* This is the implementation of the FORTRAN W(i,j,k,1)/ C style W(i,j,k,0) version from
                             * Esirkepov paper. All coordinates are rotated before thus we can
                             * always use C style W(i,j,k,0).
                             */
                            const float_X W = shapeI.DS(i) * tmp;
                            accumulated_J += W;
                            auto const atomicOp = typename T_Strategy::BlockReductionOp{};
                            atomicOp(worker, jField.template get<0>(i, j), accumulated_J);
                        }
                    }
                }

                /** Deposit current in z-direction using 2d3v model
                 *
                 * @note unlike 3d, for 2d this method has to be called explicitly
                 * with line representing whole movement of a particle on a time step (no relay point).
                 * The particle may be outside of support in x, y.
                 * T_begin and T_end must account for it, and T_ParticleAssign must work outside of support.
                 * When these conditions are met, this function is basically same to how Jz is assigned in Esirkepov 2d
                 * implementation.
                 *
                 * @param jField current density field of the particle's cell (not permuted)
                 * @param line trajectory of the virtual particle
                 * @param currentSurfaceDensityZ surface density in z direction
                 */
                template<typename T_JField, typename T_Line, typename T_Worker>
                DINLINE void computeCurrentZ(
                    T_Worker const& worker,
                    T_JField jField,
                    const T_Line& line,
                    const float_X currentSurfaceDensityZ) const
                {
                    if(currentSurfaceDensityZ == 0.0_X)
                        return;

                    auto const shapeI = makeTrajectoryAssignmentShapeFunction(
                        typename T_Strategy::template ShapeInnerLoop<T_ParticleAssign>{line.m_pos0[0], true},
                        typename T_Strategy::template ShapeInnerLoop<T_ParticleAssign>{line.m_pos1[0], true});

                    auto const shapeJ = makeTrajectoryAssignmentShapeFunction(
                        typename T_Strategy::template ShapeMiddleLoop<T_ParticleAssign>{line.m_pos0[1], true},
                        typename T_Strategy::template ShapeMiddleLoop<T_ParticleAssign>{line.m_pos1[1], true});

                    for(int j = T_begin; j < T_end; ++j)
                    {
                        const float_X s0j = shapeJ.S0(j);
                        const float_X dsj = shapeJ.S1(j) - s0j;
                        for(int i = T_begin; i < T_end; ++i)
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
            };

        } // namespace emz
    } // namespace currentSolver
} // namespace picongpu
