/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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

#include "simulation_defines.hpp"
#include "types.h"
#include "dimensions/DataSpace.hpp"
#include "basicOperations.hpp"
#include <cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include "algorithms/FieldToParticleInterpolation.hpp"
#include "algorithms/ShiftCoordinateSystem.hpp"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back_inserter.hpp>
#include "compileTime/AllCombinations.hpp"
#include "fields/currentDeposition/ZigZag/EvalAssignmentFunction.hpp"

namespace picongpu
{
namespace currentSolverZigZag
{
using namespace PMacc;

template<typename T_MathVec, typename T_Shape, typename T_Vec>
struct ShapeIt_all
{

    template<typename T_Cursor>
    HDINLINE void
    operator()(T_Cursor& cursor, const float_X F, const float3_X& pos)
    {
        typedef T_MathVec MathVec;

        const int x = MathVec::x::value;
        const int y = MathVec::y::value;
        const int z = MathVec::z::value;

        EvalAssignmentFunction<typename T_Shape::CloudShape, typename MathVec::x> AssignmentFunctionX;
        EvalAssignmentFunction< T_Shape, typename MathVec::y> AssignmentFunctionY;
        EvalAssignmentFunction< T_Shape, typename MathVec::z> AssignmentFunctionZ;

        const float_X shape_x = AssignmentFunctionX(float_X(x) - pos.x());
        const float_X shape_y = AssignmentFunctionY(float_X(y) - pos.y());
        const float_X shape_z = AssignmentFunctionZ(float_X(z) - pos.z());

        DataSpace<DIM3> jIdx;
        jIdx[T_Vec::x::value] = x;
        jIdx[T_Vec::y::value] = y;
        jIdx[T_Vec::z::value] = z;
        const float_X j = F * shape_x * shape_y*shape_z;
        PMACC_AUTO(cursorToValue,cursor(jIdx));
        atomicAddWrapper(&((*cursorToValue)[T_Vec::x::value]), j);


    }
};

/**
 * \class ZigZag charge conservation method
 * 1. order paper: "A new charge conservation method in electromagnetic particle-in-cell simulations"
 *                 by T. Umeda, Y. Omura, T. Tominaga, H. Matsumoto
 * 2. order paper: "Charge conservation methods for computing current densities in electromagnetic particle-in-cell simulations"
 *                 by T. Umeda, Y. Omura, H. Matsumoto
 * 3. order paper: "High-Order Interpolation Algorithms for Charge Conservation in Particle-in-Cell Simulation"
 *                 by Jinqing Yu, Xiaolin Jin, Weimin Zhou, Bin Li, Yuqiu Gu
 */
template<typename T_ParticleShape>
struct ZigZag<T_ParticleShape,DIM3>
{
    typedef T_ParticleShape ParticleShape;
    typedef typename ParticleShape::ChargeAssignmentOnSupport ParticleAssign;
    static const int supp = ParticleAssign::support;

    static const int currentLowerMargin = supp / 2 + 1;
    static const int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef PMacc::math::CT::Int<currentLowerMargin, currentLowerMargin, currentLowerMargin> LowerMargin;
    typedef PMacc::math::CT::Int<currentUpperMargin, currentUpperMargin, currentUpperMargin> UpperMargin;

    static const int begin = -supp / 2 + (supp + 1) % 2;
    static const int end = begin + supp;


    static const int supp_dir = supp - 1;
    static const int dir_begin = -supp_dir / 2 + (supp_dir + 1) % 2;
    static const int dir_end = dir_begin + supp_dir;

    /* begin and end border is calculated for a particle with a support which travels
     * to the negative direction.
     * Later on all coordinates shifted thus we can solve the charge calculation
     * independend from the position of the particle. That means we must not look
     * if a particle position is >0.5 oder not (this is done by coordinate shifting to this defined range)
     *
     * (supp + 1) % 2 is 1 for even supports else 0
     */

    float_X charge;

    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos1,
                            const VelType velocity,
                            const ChargeType charge, const float_X deltaTime)
    {

        const float3_X deltaPos = (velocity * deltaTime) / cellSize;

        float3_X pos[2];
        pos[0] = (pos1 - deltaPos);
        pos[1] = (pos1);

        DataSpace<DIM3> I[2];
        float3_X r;


        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < DIM3; ++d)
            {
                I[l][d] = math::floor(pos[l][d]);
            }
        }
        for (uint32_t d = 0; d < DIM3; ++d)
        {
            r[d] = calc_r(I[0][d], I[1][d], pos[0][d], pos[1][d]);
        }
        const float_X volume_reci = float_X(1.0) / float_X(CELL_VOLUME);


        for (float l = 0; l < 2; ++l)
        {
            float3_X IcP;
            float3_X F;
            const int parId = l;

            /* sign= 1 if l=0
             * sign=-1 if l=1
             */
            float_X sign = float_X(1.) - float_X(2.) * l;

            for (uint32_t d = 0; d < DIM3; ++d)
            {
                IcP[d] = calc_InCellPos(pos[parId][d], r[d], I[parId][d]);
                const float_X pos_tmp = pos[parId][d];
                const float_X r_tmp = r[d];

                F[d] = sign * calc_F(pos_tmp, r_tmp, deltaTime, charge) * volume_reci * cellSize[d];

            }


            PMACC_AUTO(cursorJ, dataBoxJ.shift(precisionCast<int>(I[parId])).toCursor());


            PMACC_AUTO( cursorJ_x,cursorJ);
            float3_X pos_x(IcP);
            typedef PMacc::math::CT::Int<supp_dir,supp,supp> Supports_x;
            ShiftCoordinateSystem<Supports_x>()(cursorJ_x, pos_x, fieldSolver::NumericalCellType::getEFieldPosition().x());
            helper<PMacc::math::CT::Int < 0, 1, 2 > >(cursorJ_x, float3_X(pos_x[0], pos_x[1], pos_x[2]), F[0]);

            float3_X pos_y = IcP;
            PMACC_AUTO( cursorJ_y,cursorJ);
            typedef PMacc::math::CT::Int<supp,supp_dir,supp> Supports_y;
            ShiftCoordinateSystem<Supports_y>()(cursorJ_y, pos_y, fieldSolver::NumericalCellType::getEFieldPosition().y());
            helper<PMacc::math::CT::Int < 1, 2, 0 > >(cursorJ_y, float3_X(pos_y[1], pos_y[2], pos_y[0]), F[1]);

            float3_X pos_z = IcP;
            PMACC_AUTO( cursorJ_z,cursorJ);
            typedef PMacc::math::CT::Int<supp,supp,supp_dir> Supports_z;
            ShiftCoordinateSystem<Supports_z>()(cursorJ_z, pos_z, fieldSolver::NumericalCellType::getEFieldPosition().z());
            helper<PMacc::math::CT::Int < 2, 0, 1 > >(cursorJ_z, float3_X(pos_z[2], pos_z[0], pos_z[1]), F[2]);
        }
    }

    template<typename T_Vec, typename CursorType>
    DINLINE void helper(CursorType& cursorJ,
                        const float3_X& pos,
                        const float_X F)
    {
        typedef boost::mpl::vector3<
            boost::mpl::range_c<int, dir_begin, dir_end >,
            boost::mpl::range_c<int, begin, end >,
            boost::mpl::range_c<int, begin, end > > Size;
        typedef typename AllCombinations<Size>::type CombiTypes;


        ForEach<CombiTypes, ShapeIt_all<bmpl::_1, ParticleShape, T_Vec> > shapeIt;
        shapeIt(forward(cursorJ), F, pos);

    }

private:

    DINLINE float_X
    calc_r(const float_X i_1, const float_X i_2, const float_X x_1, const float_X x_2) const
    {

        const float_X min_1 = ::min(i_1, i_2) + float_X(1.0);
        const float_X max_1 = ::max(i_1, i_2);
        const float_X max_2 = ::max(max_1, (x_1 + x_2) / float_X(2.));
        const float_X x_r = ::min(min_1, max_2);
        return x_r;
    }

    DINLINE float_X
    calc_InCellPos(const float_X x, const float_X x_r, const float_X i) const
    {
        return (x + x_r) / (float_X(2.0)) - i;
    }

    /* for F_2 call with -x and -x_r*/
    DINLINE float_X
    calc_F(const float_X x, const float_X x_r, const float_X& delta_t, const float_X& q) const
    {
        return q * (x_r - x) / delta_t;
    }
};

} //namespace currentSolverZigZag

} //namespace picongpu
