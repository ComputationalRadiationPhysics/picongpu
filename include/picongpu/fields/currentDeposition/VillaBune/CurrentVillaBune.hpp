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

#include <pmacc/types.hpp>
#include "picongpu/fields/currentDeposition/VillaBune/CurrentVillaBune.def"
#include "picongpu/simulation_defines.hpp"
#include "picongpu/particles/shapes/CIC.hpp"
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>

#include <type_traits>


namespace picongpu
{
    namespace currentSolver
    {
        template<typename T_ParticleShape, typename T_Strategy>
        struct VillaBune
        {
            template<class BoxJ, typename PosType, typename VelType, typename ChargeType, typename T_Acc>
            DINLINE void operator()(
                const T_Acc& acc,
                BoxJ& boxJ_par, /*box which is shifted to particles cell*/
                const PosType pos,
                const VelType velocity,
                const ChargeType charge,
                const float_X deltaTime)
            {
                /* VillaBune: field to particle interpolation _requires_ the CIC shape */
                PMACC_CASSERT_MSG_TYPE(
                    currentSolverVillaBune_requires_shapeCIC_in_particleConfig,
                    T_ParticleShape,
                    std::is_same<T_ParticleShape, particles::shapes::CIC>::value);

                // normalize deltaPos to innerCell units [0.; 1.)
                //   that means: dx_real   = v.x() * dt
                //               dx_inCell = v.x() * dt / cellSize.x()
                const float3_X deltaPos(
                    velocity.x() * deltaTime / cellSize.x(),
                    velocity.y() * deltaTime / cellSize.y(),
                    velocity.z() * deltaTime / cellSize.z());

                const PosType oldPos = (PosType)(precisionCast<float_X>(pos) - deltaPos);

                addCurrentSplitX(acc, oldPos, pos, charge, boxJ_par, deltaTime);
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                pmacc::traits::StringProperty propList("name", "VillaBune");
                return propList;
            }

        private:
            // Splits the [oldPos,newPos] beam into two beams at the x-boundary of the cell
            // if necessary

            template<typename Buffer, typename T_Acc>
            DINLINE void addCurrentSplitX(
                T_Acc const& acc,
                const float3_X& oldPos,
                const float3_X& newPos,
                const float_X charge,
                Buffer& mem,
                const float_X deltaTime)
            {
                if(pmacc::math::float2int_rd(oldPos.x()) != pmacc::math::float2int_rd(newPos.x()))
                {
                    const float3_X interPos = intersectXPlane(
                        oldPos,
                        newPos,
                        math::max(pmacc::math::float2int_rd(oldPos.x()), pmacc::math::float2int_rd(newPos.x())));
                    addCurrentSplitY(acc, oldPos, interPos, charge, mem, deltaTime);
                    addCurrentSplitY(acc, interPos, newPos, charge, mem, deltaTime);
                    return;
                }
                addCurrentSplitY(acc, oldPos, newPos, charge, mem, deltaTime);
            }

            template<typename Buffer, typename T_Acc>
            DINLINE void addCurrentToSingleCell(
                T_Acc const& acc,
                float3_X meanPos,
                const float3_X& deltaPos,
                const float_X charge,
                Buffer& memIn,
                const float_X deltaTime)
            {
                // shift to the cell meanPos belongs to
                // because meanPos may exceed the range [0,1)
                DataSpace<DIM3> off(
                    pmacc::math::float2int_rd(meanPos.x()),
                    pmacc::math::float2int_rd(meanPos.y()),
                    pmacc::math::float2int_rd(meanPos.z()));

                auto mem = memIn.shift(off);

                // fit meanPos into the range [0,1)
                meanPos.x() -= math::floor(meanPos.x());
                meanPos.y() -= math::floor(meanPos.y());
                meanPos.z() -= math::floor(meanPos.z());

                // for the formulas used in here see Villasenor/Buneman paper page 314
                const float_X tmp = deltaPos.x() * deltaPos.y() * deltaPos.z() * (float_X(1.0) / float_X(12.0));

                // j = rho * v
                //   = rho * dr / dt
                // const float_X rho = charge * (1.0 / (CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH));
                // const float_X rho_dt = rho * (1.0 / deltaTime);

                // now carefully:
                // deltaPos is in "inCell" coordinates, that means:
                //   deltaPos.x() = deltaPos_real.x() / cellSize.x()
                // to calculate the current density in realUnits it is
                //   j.x() = rho * deltaPos_real.x() / dt
                //       = rho * deltaPos.x() * cellSize.x() / dt
                // So put adding the constant directly to rho results in:
                //   const float_X rho_dtX = rho * CELL_WIDTH;
                //   const float_X rho_dtY = rho * CELL_HEIGHT;
                //   const float_X rho_dtZ = rho * CELL_DEPTH;

                // This is exactly the same like:
                // j = Q / A / t
                //   j.x() = Q.x() * (1.0 / (CELL_HEIGHT * CELL_DEPTH * deltaTime));
                //   j.y() = Q.y() * (1.0 / (CELL_WIDTH * CELL_DEPTH * deltaTime));
                //   j.z() = Q.z() * (1.0 / (CELL_WIDTH * CELL_HEIGHT * deltaTime));
                // with the difference, that (imagine a moving quader)
                //   Q.x() = charge * deltaPos_real.x() / cellsize.x()
                //       = charge * deltaPos.x() / 1.0
                //
                const float_X rho_dtX = charge * (float_X(1.0) / (CELL_HEIGHT * CELL_DEPTH * deltaTime));
                const float_X rho_dtY = charge * (float_X(1.0) / (CELL_WIDTH * CELL_DEPTH * deltaTime));
                const float_X rho_dtZ = charge * (float_X(1.0) / (CELL_WIDTH * CELL_HEIGHT * deltaTime));

                auto const atomicOp = typename T_Strategy::BlockReductionOp{};

                atomicOp(acc, mem[1][1][0].x(), rho_dtX * (deltaPos.x() * meanPos.y() * meanPos.z() + tmp));
                atomicOp(
                    acc,
                    mem[1][0][0].x(),
                    rho_dtX * (deltaPos.x() * (float_X(1.0) - meanPos.y()) * meanPos.z() - tmp));
                atomicOp(
                    acc,
                    mem[0][1][0].x(),
                    rho_dtX * (deltaPos.x() * meanPos.y() * (float_X(1.0) - meanPos.z()) - tmp));
                atomicOp(
                    acc,
                    mem[0][0][0].x(),
                    rho_dtX * (deltaPos.x() * (float_X(1.0) - meanPos.y()) * (float_X(1.0) - meanPos.z()) + tmp));

                atomicOp(acc, mem[1][0][1].y(), rho_dtY * (deltaPos.y() * meanPos.z() * meanPos.x() + tmp));
                atomicOp(
                    acc,
                    mem[0][0][1].y(),
                    rho_dtY * (deltaPos.y() * (float_X(1.0) - meanPos.z()) * meanPos.x() - tmp));
                atomicOp(
                    acc,
                    mem[1][0][0].y(),
                    rho_dtY * (deltaPos.y() * meanPos.z() * (float_X(1.0) - meanPos.x()) - tmp));
                atomicOp(
                    acc,
                    mem[0][0][0].y(),
                    rho_dtY * (deltaPos.y() * (float_X(1.0) - meanPos.z()) * (float_X(1.0) - meanPos.x()) + tmp));

                atomicOp(acc, mem[0][1][1].z(), rho_dtZ * (deltaPos.z() * meanPos.x() * meanPos.y() + tmp));
                atomicOp(
                    acc,
                    mem[0][1][0].z(),
                    rho_dtZ * (deltaPos.z() * (float_X(1.0) - meanPos.x()) * meanPos.y() - tmp));
                atomicOp(
                    acc,
                    mem[0][0][1].z(),
                    rho_dtZ * (deltaPos.z() * meanPos.x() * (float_X(1.0) - meanPos.y()) - tmp));
                atomicOp(
                    acc,
                    mem[0][0][0].z(),
                    rho_dtZ * (deltaPos.z() * (float_X(1.0) - meanPos.x()) * (float_X(1.0) - meanPos.y()) + tmp));
            }

            // calculates the intersection point of the [pos1,pos2] beam with an y,z-plane at position x0

            DINLINE float3_X intersectXPlane(const float3_X& pos1, const float3_X& pos2, const float_X x0)
            {
                const float_X t = (x0 - pos1.x()) / (pos2.x() - pos1.x());

                return float3_X(x0, pos1.y() + t * (pos2.y() - pos1.y()), pos1.z() + t * (pos2.z() - pos1.z()));
            }

            DINLINE float3_X intersectYPlane(const float3_X& pos1, const float3_X& pos2, const float_X y0)
            {
                const float_X t = (y0 - pos1.y()) / (pos2.y() - pos1.y());

                return float3_X(pos1.x() + t * (pos2.x() - pos1.x()), y0, pos1.z() + t * (pos2.z() - pos1.z()));
            }

            DINLINE float3_X intersectZPlane(const float3_X& pos1, const float3_X& pos2, const float_X z0)
            {
                const float_X t = (z0 - pos1.z()) / (pos2.z() - pos1.z());

                return float3_X(pos1.x() + t * (pos2.x() - pos1.x()), pos1.y() + t * (pos2.y() - pos1.y()), z0);
            }

            // Splits the [oldPos,newPos] beam into two beams at the z-boundary of the cell
            // if necessary

            template<typename Buffer, typename T_Acc>
            DINLINE void addCurrentSplitZ(
                T_Acc const& acc,
                const float3_X& oldPos,
                const float3_X& newPos,
                const float_X charge,
                Buffer& mem,
                const float_X deltaTime)
            {
                if(pmacc::math::float2int_rd(oldPos.z()) != pmacc::math::float2int_rd(newPos.z()))
                {
                    const float3_X interPos = intersectZPlane(
                        oldPos,
                        newPos,
                        math::max(pmacc::math::float2int_rd(oldPos.z()), pmacc::math::float2int_rd(newPos.z())));
                    float3_X deltaPos = interPos - oldPos;
                    float3_X meanPos = oldPos + float_X(0.5) * deltaPos;
                    addCurrentToSingleCell(acc, meanPos, deltaPos, charge, mem, deltaTime);

                    deltaPos = newPos - interPos;
                    meanPos = interPos + float_X(0.5) * deltaPos;
                    addCurrentToSingleCell(acc, meanPos, deltaPos, charge, mem, deltaTime);
                    return;
                }
                const float3_X deltaPos = newPos - oldPos;
                const float3_X meanPos = oldPos + float_X(0.5) * deltaPos;
                addCurrentToSingleCell(acc, meanPos, deltaPos, charge, mem, deltaTime);
            }

            // Splits the [oldPos,newPos] beam into two beams at the y-boundary of the cell
            // if necessary

            template<typename Buffer, typename T_Acc>
            DINLINE void addCurrentSplitY(
                T_Acc const& acc,
                const float3_X& oldPos,
                const float3_X& newPos,
                const float_X charge,
                Buffer& mem,
                const float_X deltaTime)
            {
                if(pmacc::math::float2int_rd(oldPos.y()) != pmacc::math::float2int_rd(newPos.y()))
                {
                    const float3_X interPos = intersectYPlane(
                        oldPos,
                        newPos,
                        math::max(pmacc::math::float2int_rd(oldPos.y()), pmacc::math::float2int_rd(newPos.y())));
                    addCurrentSplitZ(acc, oldPos, interPos, charge, mem, deltaTime);
                    addCurrentSplitZ(acc, interPos, newPos, charge, mem, deltaTime);
                    return;
                }
                addCurrentSplitZ(acc, oldPos, newPos, charge, mem, deltaTime);
            }
        };

    } // namespace currentSolver

    namespace traits
    {
        template<typename T_ParticleShape, typename T_Strategy>
        struct GetMargin<picongpu::currentSolver::VillaBune<T_ParticleShape, T_Strategy>>
        {
            using LowerMargin = ::pmacc::math::CT::Int<1, 1, 1>;
            using UpperMargin = ::pmacc::math::CT::Int<2, 2, 2>;

            /** maximum margin size of LowerMargin and UpperMargin */
            static constexpr int maxMargin = 2;

            PMACC_CASSERT_MSG(
                __VillaBune_supercell_or_number_of_guard_supercells_is_too_small_for_stencil,
                pmacc::math::CT::min<typename pmacc::math::CT::mul<SuperCellSize, GuardSize>::type>::type::value
                    >= maxMargin);
        };

    } // namespace traits

} // namespace picongpu
