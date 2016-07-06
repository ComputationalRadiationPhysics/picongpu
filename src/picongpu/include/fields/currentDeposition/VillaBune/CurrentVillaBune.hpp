/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera
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

#include "pmacc_types.hpp"
#include "simulation_defines.hpp"
#include "dimensions/DataSpace.hpp"
#include "basicOperations.hpp"
#include "math/Vector.hpp"
#include "traits/IsSameType.hpp"
#include "particles/shapes/CIC.hpp"

namespace picongpu
{
namespace currentSolver
{
using namespace PMacc;

template<typename T_ParticleShape>
struct VillaBune
{
    template<class BoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(BoxJ& boxJ_par, /*box which is shifted to particles cell*/
                            const PosType pos,
                            const VelType velocity,
                            const ChargeType charge, const float_X deltaTime)
    {
        /* VillaBune: field to particle interpolation _requires_ the CIC shape */
        PMACC_CASSERT_MSG_TYPE(currentSolverVillaBune_requires_shapeCIC_in_particleConfig,
                    T_ParticleShape,
                    T_ParticleShape::support == 2);

        // normalize deltaPos to innerCell units [0.; 1.)
        //   that means: dx_real   = v.x() * dt
        //               dx_inCell = v.x() * dt / cellSize.x()
        const float3_X deltaPos(
                                velocity.x() * deltaTime / cellSize.x(),
                                velocity.y() * deltaTime / cellSize.y(),
                                velocity.z() * deltaTime / cellSize.z());

        const PosType oldPos = (PosType) (precisionCast<float_X > (pos) - deltaPos);

        addCurrentSplitX(oldPos, pos, charge, boxJ_par, deltaTime);
    }

    static PMacc::traits::StringProperty getStringProperties()
    {
        PMacc::traits::StringProperty propList( "name", "VillaBune" );
        return propList;
    }

private:
    //Splits the [oldPos,newPos] beam into two beams at the x-boundary of the cell
    //if necessary

    template<class Buffer >
    DINLINE void addCurrentSplitX(const float3_X& oldPos, const float3_X& newPos,
                                  const float_X charge, Buffer & mem, const float_X deltaTime)
    {

        if (math::float2int_rd(oldPos.x()) != math::float2int_rd(newPos.x()))
        {
            const float3_X interPos = intersectXPlane(oldPos, newPos,
                                                      max(math::float2int_rd(oldPos.x()), math::float2int_rd(newPos.x())));
            addCurrentSplitY(oldPos, interPos, charge, mem, deltaTime);
            addCurrentSplitY(interPos, newPos, charge, mem, deltaTime);
            return;
        }
        addCurrentSplitY(oldPos, newPos, charge, mem, deltaTime);
    }

    template<class Buffer >
    DINLINE void addCurrentToSingleCell(float3_X meanPos, const float3_X& deltaPos,
                                        const float_X charge, Buffer & memIn, const float_X deltaTime)
    {
        //shift to the cell meanPos belongs to
        //because meanPos may exceed the range [0,1)
        DataSpace<DIM3> off(math::float2int_rd(meanPos.x()),
                            math::float2int_rd(meanPos.y()),
                            math::float2int_rd(meanPos.z()));

        PMACC_AUTO(mem, memIn.shift(off));

        //fit meanPos into the range [0,1)
        meanPos.x() -= math::floor(meanPos.x());
        meanPos.y() -= math::floor(meanPos.y());
        meanPos.z() -= math::floor(meanPos.z());

        //for the formulas used in here see Villasenor/Buneman paper page 314
        const float_X tmp = deltaPos.x() * deltaPos.y() * deltaPos.z() * (float_X(1.0) / float_X(12.0));

        // j = rho * v
        //   = rho * dr / dt
        //const float_X rho = charge * (1.0 / (CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH));
        //const float_X rho_dt = rho * (1.0 / deltaTime);

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

        atomicAddWrapper(&(mem[1][1][0].x()), rho_dtX * (deltaPos.x() * meanPos.y() * meanPos.z() + tmp));
        atomicAddWrapper(&(mem[1][0][0].x()), rho_dtX * (deltaPos.x() * (float_X(1.0) - meanPos.y()) * meanPos.z() - tmp));
        atomicAddWrapper(&(mem[0][1][0].x()), rho_dtX * (deltaPos.x() * meanPos.y() * (float_X(1.0) - meanPos.z()) - tmp));
        atomicAddWrapper(&(mem[0][0][0].x()), rho_dtX * (deltaPos.x() * (float_X(1.0) - meanPos.y()) * (float_X(1.0) - meanPos.z()) + tmp));

        atomicAddWrapper(&(mem[1][0][1].y()), rho_dtY * (deltaPos.y() * meanPos.z() * meanPos.x() + tmp));
        atomicAddWrapper(&(mem[0][0][1].y()), rho_dtY * (deltaPos.y() * (float_X(1.0) - meanPos.z()) * meanPos.x() - tmp));
        atomicAddWrapper(&(mem[1][0][0].y()), rho_dtY * (deltaPos.y() * meanPos.z() * (float_X(1.0) - meanPos.x()) - tmp));
        atomicAddWrapper(&(mem[0][0][0].y()), rho_dtY * (deltaPos.y() * (float_X(1.0) - meanPos.z()) * (float_X(1.0) - meanPos.x()) + tmp));

        atomicAddWrapper(&(mem[0][1][1].z()), rho_dtZ * (deltaPos.z() * meanPos.x() * meanPos.y() + tmp));
        atomicAddWrapper(&(mem[0][1][0].z()), rho_dtZ * (deltaPos.z() * (float_X(1.0) - meanPos.x()) * meanPos.y() - tmp));
        atomicAddWrapper(&(mem[0][0][1].z()), rho_dtZ * (deltaPos.z() * meanPos.x() * (float_X(1.0) - meanPos.y()) - tmp));
        atomicAddWrapper(&(mem[0][0][0].z()), rho_dtZ * (deltaPos.z() * (float_X(1.0) - meanPos.x()) * (float_X(1.0) - meanPos.y()) + tmp));

    }

    //calculates the intersection point of the [pos1,pos2] beam with an y,z-plane at position x0

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

    //Splits the [oldPos,newPos] beam into two beams at the z-boundary of the cell
    //if necessary

    template<class Buffer >
    DINLINE void addCurrentSplitZ(const float3_X &oldPos, const float3_X &newPos,
                                  const float_X charge, Buffer & mem, const float_X deltaTime)
    {

        if (math::float2int_rd(oldPos.z()) != math::float2int_rd(newPos.z()))
        {
            const float3_X interPos = intersectZPlane(oldPos, newPos,
                                                      max(math::float2int_rd(oldPos.z()), math::float2int_rd(newPos.z())));
            float3_X deltaPos = interPos - oldPos;
            float3_X meanPos = oldPos + float_X(0.5) * deltaPos;
            addCurrentToSingleCell(meanPos, deltaPos, charge, mem, deltaTime);

            deltaPos = newPos - interPos;
            meanPos = interPos + float_X(0.5) * deltaPos;
            addCurrentToSingleCell(meanPos, deltaPos, charge, mem, deltaTime);
            return;
        }
        const float3_X deltaPos = newPos - oldPos;
        const float3_X meanPos = oldPos + float_X(0.5) * deltaPos;
        addCurrentToSingleCell(meanPos, deltaPos, charge, mem, deltaTime);
    }

    //Splits the [oldPos,newPos] beam into two beams at the y-boundary of the cell
    //if necessary

    template<class Buffer >
    DINLINE void addCurrentSplitY(const float3_X& oldPos, const float3_X& newPos,
                                  const float_X charge, Buffer & mem, const float_X deltaTime)
    {

        if (math::float2int_rd(oldPos.y()) != math::float2int_rd(newPos.y()))
        {
            const float3_X interPos = intersectYPlane(oldPos, newPos,
                                                      max(math::float2int_rd(oldPos.y()), math::float2int_rd(newPos.y())));
            addCurrentSplitZ(oldPos, interPos, charge, mem, deltaTime);
            addCurrentSplitZ(interPos, newPos, charge, mem, deltaTime);
            return;
        }
        addCurrentSplitZ(oldPos, newPos, charge, mem, deltaTime);
    }

};

} //namespace currentSolver

namespace traits
{

template<typename T_ParticleShape>
struct GetMargin<picongpu::currentSolver::VillaBune<T_ParticleShape> >
{
    typedef ::PMacc::math::CT::Int < 1, 1, 1 > LowerMargin;
    typedef ::PMacc::math::CT::Int < 2, 2, 2 > UpperMargin;
};

} //namespace traits

} //namespace picongpu



