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
struct AssignChargeToCell
{

    template<typename T_Cursor>
    HDINLINE void
    operator()(T_Cursor& cursor, const float3_X& pos, const float_X F)
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
        PMACC_AUTO(cursorToValue, cursor(jIdx));
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
struct ZigZag<T_ParticleShape, DIM3>
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

    template<typename T_Swivel>
    struct AssignOneDirection
    {

        template<typename T_Cursor>
        HDINLINE void
        operator()(T_Cursor cursor, float3_X pos, const float3_X& F)
        {
            const uint32_t dir = T_Swivel::x::value;

            typedef PMacc::math::CT::Int<supp, supp, supp> Supports_full;
            typedef typename PMacc::math::CT::Assign<
                typename Supports_full::This,
                bmpl::integral_c<uint32_t, dir>,
                bmpl::integral_c<int, supp_dir> >::type Supports_direction;

            /* shift coordinate system to
             *   - support different numerical cell types
             *   - run calculations in a shape optimized coordinate system
             *     with fixed interpolation points
             */
            ShiftCoordinateSystem<Supports_direction>()(cursor, pos, fieldSolver::NumericalCellType::getEFieldPosition()[dir]);

            const float3_X pos_dir(pos[T_Swivel::x::value],
                                   pos[T_Swivel::y::value],
                                   pos[T_Swivel::z::value]);

            typedef boost::mpl::vector3<
                boost::mpl::range_c<int, dir_begin, dir_end >,
                boost::mpl::range_c<int, begin, end >,
                boost::mpl::range_c<int, begin, end > > Size;
            typedef typename AllCombinations<Size>::type CombiTypes;

            ForEach<CombiTypes, AssignChargeToCell<bmpl::_1, ParticleShape, T_Swivel> > callAssignChargeToCell;
            callAssignChargeToCell(forward(cursor), pos_dir, F[dir]);
        }

    };

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

            typedef bmpl::vector3<
                PMacc::math::CT::Int < 0, 1, 2 >,
                PMacc::math::CT::Int < 1, 2, 0 >,
                PMacc::math::CT::Int < 2, 0, 1 > > Directions;

            ForEach<Directions, AssignOneDirection<bmpl::_1> > callAssignOneDirection;
            callAssignOneDirection(forward(cursorJ), IcP, F);
        }
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
