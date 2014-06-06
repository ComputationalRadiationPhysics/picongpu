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
#include <boost/mpl/insert.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/if.hpp>
#include "compileTime/AllCombinations.hpp"
#include "fields/currentDeposition/ZigZag/EvalAssignmentFunction.hpp"

namespace picongpu
{
namespace currentSolver
{
using namespace PMacc;

template<typename T_ShapeComponent, typename T_CurrentDirection, typename T_GridPointVec, typename T_Shape>
struct EvalAssignmentFunctionOfDirection
{

    HDINLINE void
    operator()(float_X& result, const floatD_X& pos)
    {
        typedef T_GridPointVec GridPointVec;
        typedef T_ShapeComponent ShapeComponent;
        typedef T_CurrentDirection CurrentDirection;

        const int component = ShapeComponent::value;

        typedef typename bmpl::if_ <
            bmpl::equal_to<ShapeComponent, CurrentDirection>,
            typename T_Shape::CloudShape,
            T_Shape
            >::type Shape;

        typedef typename GridPointVec::template at<component>::type GridPoint;
        currentSolverZigZag::EvalAssignmentFunction< Shape, GridPoint > AssignmentFunction;

        const float_X gridPoint = GridPoint::value;
        const float_X shape_value = AssignmentFunction(gridPoint - pos[component]);
        result *= shape_value;
    }
};

template<typename T_GridPointVec, typename T_Shape, typename T_CurrentComponent>
struct AssignChargeToCell
{

    template<typename T_Cursor>
    HDINLINE void
    operator()(T_Cursor& cursor, const floatD_X& pos, const float_X F)
    {
        typedef T_GridPointVec GridPointVec;
        typedef T_Shape Shape;
        typedef T_CurrentComponent CurrentComponent;

        typedef boost::mpl::range_c<int, 0, simDim > ShapeComponentsRange;
        /* this transformation is needed to use boost::ml::accumulate on ComponentsRange*/
        typedef typename MakeSeq< ShapeComponentsRange>::type ShapeComponents;

        ForEach<ShapeComponents,
            EvalAssignmentFunctionOfDirection<bmpl::_1, CurrentComponent, GridPointVec, Shape>
            > evalShape;
        float_X j = F;
        evalShape(forward(j), pos);

        const uint32_t currentComponent = CurrentComponent::value;

        PMACC_AUTO(cursorToValue, cursor(GridPointVec::toRT()));
        atomicAddWrapper(&((*cursorToValue)[currentComponent]), j);
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
struct ZigZag
{
    typedef T_ParticleShape ParticleShape;
    typedef typename ParticleShape::ChargeAssignmentOnSupport ParticleAssign;
    static const int supp = ParticleAssign::support;

    static const int currentLowerMargin = supp / 2 + 1;
    static const int currentUpperMargin = (supp + 1) / 2 + 1;
    typedef typename PMacc::math::CT::make_Int<simDim, currentLowerMargin>::type LowerMargin;
    typedef typename PMacc::math::CT::make_Int<simDim, currentUpperMargin>::type UpperMargin;

    static const int begin = -supp / 2 + (supp + 1) % 2;
    static const int end = begin + supp;


    static const int supp_dir = supp - 1;
    static const int dir_begin = -supp_dir / 2 + (supp_dir + 1) % 2;
    static const int dir_end = dir_begin + supp_dir;

    template<typename T_CurrentComponent>
    struct AssignOneDirection
    {

        template<typename T_Cursor>
        HDINLINE void
        operator()(T_Cursor cursor, floatD_X pos, const float3_X& flux)
        {
            typedef T_CurrentComponent CurrentComponent;
            const uint32_t dir = CurrentComponent::value;

            typedef typename PMacc::math::CT::make_Int<simDim, supp>::type Supports_full;
            typedef typename PMacc::math::CT::AssignIfInRange<
                typename Supports_full::This,
                bmpl::integral_c<uint32_t, dir>,
                bmpl::integral_c<int, supp_dir> >::type Supports_direction;

            /* shift coordinate system to
             *   - support different numerical cell types
             *   - run calculations in a shape optimized coordinate system
             *     with fixed interpolation points
             */
            ShiftCoordinateSystem<Supports_direction>()(cursor, pos, fieldSolver::NumericalCellType::getEFieldPosition()[dir]);

            typedef typename PMacc::math::CT::make_Vector<
                simDim,
                boost::mpl::range_c<int, begin, end > >::type Size_full;

            typedef typename PMacc::math::CT::AssignIfInRange<
                typename Size_full::This,
                bmpl::integral_c<uint32_t, dir>,
                boost::mpl::range_c<int, dir_begin, dir_end > >::type::mplVector Size;


            typedef typename AllCombinations<Size>::type CombiTypes;
            ForEach<CombiTypes, AssignChargeToCell<bmpl::_1, ParticleShape, CurrentComponent > > callAssignChargeToCell;
            callAssignChargeToCell(forward(cursor), pos, flux[dir]);
        }

    };

    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType >
    DINLINE void operator()(DataBoxJ dataBoxJ,
                            const PosType pos1,
                            const VelType velocity,
                            const ChargeType charge, const float_X deltaTime)
    {

        floatD_X deltaPos;
        for (uint32_t d = 0; d < simDim; ++d)
            deltaPos[d] = (velocity[d] * deltaTime) / cellSize[d];

        floatD_X pos[2];
        pos[0] = (pos1 - deltaPos);
        pos[1] = (pos1);

        DataSpace<simDim> I[2];
        floatD_X r;


        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < simDim; ++d)
            {
                I[l][d] = math::floor(pos[l][d]);
            }
        }
        for (uint32_t d = 0; d < simDim; ++d)
        {
            r[d] = calc_r(I[0][d], I[1][d], pos[0][d], pos[1][d]);
        }
        const float_X volume_reci = float_X(1.0) / float_X(CELL_VOLUME);


        for (float l = 0; l < 2; ++l)
        {
            floatD_X inCellPos;
            float3_X flux;
            const int parId = l;

            /* sign= 1 if l=0
             * sign=-1 if l=1
             */
            float_X sign = float_X(1.) - float_X(2.) * l;
            for (uint32_t d = 0; d < simDim; ++d)
            {
                const float_X pos_tmp = pos[parId][d];
                const float_X r_tmp = r[d];
                inCellPos[d] = calc_InCellPos(pos_tmp, r_tmp, I[parId][d]);
                flux[d] = sign * calc_F(pos_tmp, r_tmp, deltaTime, charge) * volume_reci * cellSize[d];

            }
            for (uint32_t d = simDim; d < 3; ++d)
            {
                flux[d] = charge * velocity[d] * volume_reci;
            }

            PMACC_AUTO(cursorJ, dataBoxJ.shift(precisionCast<int>(I[parId])).toCursor());

            typedef boost::mpl::range_c<int, 0, 3 > ComponentsRange;
            /* this transformation is needed to use boost::ml::accumulate on ComponentsRange*/
            typedef typename MakeSeq<ComponentsRange>::type Components;

            ForEach<Components, AssignOneDirection<bmpl::_1> > callAssignOneDirection;
            callAssignOneDirection(forward(cursorJ), inCellPos, flux);
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

} //namespace currentSolver

} //namespace picongpu
