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

#include <pmacc/cuSTL/cursor/Cursor.hpp>
#include <pmacc/cuSTL/cursor/tools/twistVectorFieldAxes.hpp>

#include "picongpu/fields/currentDeposition/EmZ/EmZ.def"
#include "picongpu/fields/currentDeposition/Esirkepov/Line.hpp"


namespace picongpu
{
    namespace currentSolver
    {
        namespace emz
        {
            using namespace pmacc;

            template<typename ParticleAssign>
            struct BaseMethods
            {
                /** evaluate particle shape
                 * @param line element with previous and current position of the particle
                 * @param gridPoint used grid point to evaluate assignment shape
                 * @param d dimension range {0,1,2} means {x,y,z}
                 *          different to Esirkepov paper, here we use C style
                 * @{
                 */

                /** evaluate shape for the first particle S0 (see paper) */
                DINLINE float_X S0(const Line<floatD_X>& line, const float_X gridPoint, const uint32_t d) const
                {
                    return ParticleAssign()(gridPoint - line.m_pos0[d]);
                }

                /** evaluate shape for the second particle */
                DINLINE float_X S1(const Line<floatD_X>& line, const float_X gridPoint, const uint32_t d) const
                {
                    return ParticleAssign()(gridPoint - line.m_pos1[d]);
                }
                /*! @} */

                /** calculate DS (see paper)
                 * @param line element with previous and current position of the particle
                 * @param gridPoint used grid point to evaluate assignment shape
                 * @param d dimension range {0,1,2} means {x,y,z}]
                 *          different to Esirkepov paper, here we use C style
                 */
                DINLINE float_X DS(const Line<floatD_X>& line, const float_X gridPoint, const uint32_t d) const
                {
                    return ParticleAssign()(gridPoint - line.m_pos1[d]) - ParticleAssign()(gridPoint - line.m_pos0[d]);
                }
            };

            template<typename T_AtomicAddOp, typename ParticleAssign, int T_begin, int T_end>
            struct DepositCurrent<T_AtomicAddOp, ParticleAssign, T_begin, T_end, DIM3>
                : public BaseMethods<ParticleAssign>
            {
                template<typename T_Cursor, typename T_Acc>
                DINLINE void operator()(
                    T_Acc const& acc,
                    const T_Cursor& cursorJ,
                    const Line<float3_X>& line,
                    const float_X chargeDensity,
                    const float_X) const
                {
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
                        cellSize.x() * chargeDensity / DELTA_T);
                    cptCurrent1D(
                        acc,
                        twistVectorFieldAxes<pmacc::math::CT::Int<2, 0, 1>>(cursorJ),
                        rotateOrigin<2, 0, 1>(line),
                        cellSize.y() * chargeDensity / DELTA_T);
                    cptCurrent1D(acc, cursorJ, line, cellSize.z() * chargeDensity / DELTA_T);
                }

                /** deposites current in z-direction
                 *
                 * \param cursorJ cursor pointing at the current density field of the particle's cell
                 * \param line trajectory of the virtual particle
                 * \param currentSurfaceDensity surface density
                 */
                template<typename CursorJ, typename T_Line, typename T_Acc>
                DINLINE void cptCurrent1D(
                    T_Acc const& acc,
                    CursorJ cursorJ,
                    const T_Line& line,
                    const float_X currentSurfaceDensity) const
                {
                    if(line.m_pos0[2] == line.m_pos1[2])
                        return;
                    /* pick every cell in the xy-plane that is overlapped by particle's
                     * form factor and deposit the current for the cells above and beneath
                     * that cell and for the cell itself.
                     */
                    for(int i = T_begin; i < T_end; ++i)
                    {
                        const float_X s0i = this->S0(line, i, 0);
                        const float_X dsi = this->S1(line, i, 0) - s0i;
                        for(int j = T_begin; j < T_end; ++j)
                        {
                            const float_X s0j = this->S0(line, j, 1);
                            const float_X dsj = this->S1(line, j, 1) - s0j;

                            float_X tmp = -currentSurfaceDensity
                                * (s0i * s0j + float_X(0.5) * (dsi * s0j + s0i * dsj)
                                   + (float_X(1.0) / float_X(3.0)) * dsj * dsi);

                            float_X accumulated_J = float_X(0.0);
                            for(int k = T_begin; k < T_end - 1; ++k)
                            {
                                /* This is the implementation of the FORTRAN W(i,j,k,3)/ C style W(i,j,k,2) version
                                 * from Esirkepov paper. All coordinates are rotated before thus we can always use C
                                 * style W(i,j,k,2).
                                 */
                                const float_X W = this->DS(line, k, 2) * tmp;
                                accumulated_J += W;
                                auto const atomicOp = T_AtomicAddOp{};
                                atomicOp(acc, (*cursorJ(i, j, k)).z(), accumulated_J);
                            }
                        }
                    }
                }
            };

            template<typename T_AtomicAddOp, typename ParticleAssign, int T_begin, int T_end>
            struct DepositCurrent<T_AtomicAddOp, ParticleAssign, T_begin, T_end, DIM2>
                : public BaseMethods<ParticleAssign>
            {
                template<typename T_Cursor, typename T_Acc>
                DINLINE void operator()(
                    T_Acc const& acc,
                    const T_Cursor& cursorJ,
                    const Line<float2_X>& line,
                    const float_X chargeDensity,
                    const float_X velocityZ) const
                {
                    using namespace cursor::tools;
                    cptCurrent1D(acc, cursorJ, line, cellSize.x() * chargeDensity / DELTA_T);
                    cptCurrent1D(
                        acc,
                        twistVectorFieldAxes<pmacc::math::CT::Int<1, 0>>(cursorJ),
                        rotateOrigin<1, 0>(line),
                        cellSize.y() * chargeDensity / DELTA_T);
                    cptCurrentZ(acc, cursorJ, line, velocityZ * chargeDensity);
                }

                /** deposites current in x-direction
                 *
                 * \param cursorJ cursor pointing at the current density field of the particle's cell
                 * \param line trajectory of the virtual particle
                 * \param currentSurfaceDensity surface density
                 */
                template<typename CursorJ, typename T_Line, typename T_Acc>
                DINLINE void cptCurrent1D(
                    T_Acc const& acc,
                    CursorJ cursorJ,
                    const T_Line& line,
                    const float_X currentSurfaceDensity) const
                {
                    if(line.m_pos0[0] == line.m_pos1[0])
                        return;

                    for(int j = T_begin; j < T_end; ++j)
                    {
                        const float_X s0j = this->S0(line, j, 1);
                        const float_X dsj = this->S1(line, j, 1) - s0j;

                        float_X tmp = -currentSurfaceDensity * (s0j + float_X(0.5) * dsj);

                        float_X accumulated_J = float_X(0.0);
                        for(int i = T_begin; i < T_end - 1; ++i)
                        {
                            /* This is the implementation of the FORTRAN W(i,j,k,1)/ C style W(i,j,k,0) version from
                             * Esirkepov paper. All coordinates are rotated before thus we can
                             * always use C style W(i,j,k,0).
                             */
                            const float_X W = this->DS(line, i, 0) * tmp;
                            accumulated_J += W;
                            auto const atomicOp = T_AtomicAddOp{};
                            atomicOp(acc, (*cursorJ(i, j)).x(), accumulated_J);
                        }
                    }
                }

                /** deposites current in z-direction
                 *
                 * \param cursorJ cursor pointing at the current density field of the particle's cell
                 * \param line trajectory of the virtual particle
                 * \param currentSurfaceDensityZ surface density in z direction
                 */
                template<typename CursorJ, typename T_Line, typename T_Acc>
                DINLINE void cptCurrentZ(
                    T_Acc const& acc,
                    CursorJ cursorJ,
                    const T_Line& line,
                    const float_X currentSurfaceDensityZ) const
                {
                    if(currentSurfaceDensityZ == float_X(0.0))
                        return;

                    for(int j = T_begin; j < T_end; ++j)
                    {
                        const float_X s0j = this->S0(line, j, 1);
                        const float_X dsj = this->S1(line, j, 1) - s0j;
                        for(int i = T_begin; i < T_end; ++i)
                        {
                            const float_X s0i = this->S0(line, i, 0);
                            const float_X dsi = this->S1(line, i, 0) - s0i;
                            float_X W = s0i * this->S0(line, j, 1) + float_X(0.5) * (dsi * s0j + s0i * dsj)
                                + (float_X(1.0) / float_X(3.0)) * dsi * dsj;

                            const float_X j_z = W * currentSurfaceDensityZ;
                            auto const atomicOp = T_AtomicAddOp{};
                            atomicOp(acc, (*cursorJ(i, j)).z(), j_z);
                        }
                    }
                }
            };

        } // namespace emz
    } // namespace currentSolver
} // namespace picongpu
