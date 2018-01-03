/* Copyright 2013-2018 Axel Huebl, Heiko Burau, Rene Widera
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
#include <pmacc/cuSTL/cursor/tools/twistVectorFieldAxes.hpp>
#include "picongpu/algorithms/FieldToParticleInterpolation.hpp"
#include "picongpu/algorithms/ShiftCoordinateSystem.hpp"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/if.hpp>
#include <pmacc/compileTime/AllCombinations.hpp>
#include "picongpu/fields/currentDeposition/ZigZag/EvalAssignmentFunction.hpp"


namespace picongpu
{
namespace currentSolver
{
using namespace pmacc;

/** functor to get shape factor
 *
 * Calculate: AssignmentShape(grid point - particle point)
 *
 * AssignmentShape depends on the calculation direction and the current component
 * If current component is X and calculation direction is also X than particle cloud shape is used.
 *
 * @tparam T_ShapeComponent integral type in which direction the shape should be evaluated
 * @tparam T_CurrentComponent integral type with component information
 * @tparam T_GridPointVec integral type which define grid point
 * @tparam T_Shape assignment shape of the particle
 */
template<typename T_ShapeComponent, typename T_CurrentDirection, typename T_GridPointVec, typename T_Shape>
struct EvalAssignmentFunctionOfDirection
{

    HDINLINE void
    operator()(float_X& result, const floatD_X& pos)
    {
        using GridPointVec = T_GridPointVec;
        using ShapeComponent = T_ShapeComponent;
        using CurrentDirection = T_CurrentDirection;

        const int component = ShapeComponent::value;

        /* select assignment shape
         * if component is equal to direction we use particle cloud shape
         * else
         * particle assignment shape
         */
        using Shape = typename bmpl::if_<
            bmpl::equal_to<
                ShapeComponent,
                CurrentDirection
            >,
            typename T_Shape::CloudShape,
            T_Shape
        >::type;

        using GridPoint = typename GridPointVec::template at< component >::type;
        currentSolverZigZag::EvalAssignmentFunction< Shape, GridPoint > AssignmentFunction;

        /* calculate assignment factor*/
        const float_X shape_value = AssignmentFunction(pos[component]);
        result *= shape_value;
    }
};

/** functor to calculate current for one cell (grid point)
 *
 * @tparam T_GridPointVec integral type which define grid point
 * @tparam T_Shape assignment shape of the particle
 * @tparam T_CurrentComponent integral type with component information
 */
template<typename T_GridPointVec, typename T_Shape, typename T_CurrentComponent>
struct AssignChargeToCell
{

    template<
        typename T_Cursor,
        typename T_Acc
    >
    HDINLINE void
    operator()(
        T_Acc const & acc,
        T_Cursor& cursor,
        const floatD_X& pos,
        const float_X flux
    )
    {
        using GridPointVec = T_GridPointVec;
        using Shape = T_Shape;
        using CurrentComponent = T_CurrentComponent;

        // evaluate shape in direction [0;simDim)
        using ShapeComponentsRange = boost::mpl::range_c< int, 0, simDim >;
        // this transformation is needed to use boost::ml::accumulate on ComponentsRange
        using ShapeComponents = typename MakeSeq< ShapeComponentsRange >::type;

        ForEach<ShapeComponents,
            EvalAssignmentFunctionOfDirection<bmpl::_1, CurrentComponent, GridPointVec, Shape>
            > evalShape;
        float_X j = flux;
        /* N=simDim-1
         * calculate j=flux*Shape_0(pos)*...*SHAPE_N(pos) */
        evalShape(forward(j), pos);

        const uint32_t currentComponent = CurrentComponent::value;

        /* shift memory cursor to cell (grid point)*/
        auto cursorToValue = cursor(GridPointVec::toRT());
        /* add current to component of the cell*/
        atomicAdd(&((*cursorToValue)[currentComponent]), j, ::alpaka::hierarchy::Threads{});
    }
};

/** ZigZag charge conservation method
 *
 * @see ZigZag.def for paper references
 */
template<typename T_ParticleShape>
struct ZigZag
{
    /* cloud shape: describe the form factor of a particle
     * assignment shape: integral over the cloud shape (this shape is defined by the user in
     * species.param for a species)
     */
    using ParticleShape = T_ParticleShape;
    using ParticleAssign = typename ParticleShape::ChargeAssignmentOnSupport;
    static constexpr int supp = ParticleAssign::support;

    static constexpr int currentLowerMargin = supp / 2 + 1;
    static constexpr int currentUpperMargin = (supp + 1) / 2 + 1;
    using LowerMargin = typename pmacc::math::CT::make_Int< simDim, currentLowerMargin >::type;
    using UpperMargin = typename pmacc::math::CT::make_Int< simDim, currentUpperMargin >::type;

    PMACC_CASSERT_MSG(
        __ZigZag_supercell_is_to_small_for_stencil,
        pmacc::math::CT::min< SuperCellSize >::type::value >= currentLowerMargin &&
        pmacc::math::CT::min< SuperCellSize >::type::value >= currentUpperMargin
    );

    /* calculate grid point where we calculate the assigned values
     * grid points are independent of particle position if we use
     * @see ShiftCoordinateSystem
     * grid points were we calculate the current [begin;end)
     */
    static constexpr int begin = -supp / 2 + (supp + 1) % 2;
    static constexpr int end = begin + supp;

    /* same as begin and end but for the direction where we calculate j
     * supp_dir = support of the cloud shape
     */
    static constexpr int supp_dir = supp - 1;
    static constexpr int dir_begin = -supp_dir / 2 + (supp_dir + 1) % 2;
    static constexpr int dir_end = dir_begin + supp_dir;

    /** functor to calculate current for one direction
     *
     * @tparam T_CurrentComponent integral type with component information
     * (x=0; y=1; z=2)
     */
    template<typename T_CurrentComponent>
    struct AssignOneDirection
    {

        template<
            typename T_Cursor,
            typename T_Acc
        >
        HDINLINE void
        operator()(
            T_Acc const & acc,
            T_Cursor cursor,
            floatD_X pos,
            const float3_X& flux
        )
        {
            using CurrentComponent = T_CurrentComponent;
            const uint32_t dir = CurrentComponent::value;

            /* if the flux is zero there is no need to deposit any current */
            if (flux[dir] == float_X(0.0))
                return;
            /* create support information to shift our coordinate system
             * use support of the particle assignment function
             */
            using Supports_full = typename pmacc::math::CT::make_Int< simDim, supp >::type;
            // set evaluation direction to the support of the cloud particle shape function
            using Supports_direction = typename pmacc::math::CT::AssignIfInRange<
                typename Supports_full::This,
                bmpl::integral_c< uint32_t, dir >,
                bmpl::integral_c< int, supp_dir >
            >::type;

            /* shift coordinate system to
             *   - support different numerical cell types
             *   - run calculations in a shape optimized coordinate system
             *     with fixed interpolation points
             */
            const fieldSolver::numericalCellType::traits::FieldPosition<FieldJ> fieldPosJ;
            ShiftCoordinateSystem<Supports_direction>()(cursor, pos, fieldPosJ()[dir]);

            /* define grid points where we evaluate the shape function*/
            using Size_full = typename pmacc::math::CT::make_Vector<
                simDim,
                boost::mpl::range_c< int, begin, end >
            >::type;

            /* set grid points for the evaluation direction*/
            using Size = typename pmacc::math::CT::AssignIfInRange<
                typename Size_full::This,
                bmpl::integral_c< uint32_t, dir >,
                boost::mpl::range_c< int, dir_begin, dir_end >
            >::type::mplVector;

            /* calculate the current for every cell (grid point)*/
            using CombiTypes = typename AllCombinations<Size>::type;
            ForEach<
                CombiTypes,
                AssignChargeToCell<
                    bmpl::_1,
                    ParticleShape,
                    CurrentComponent
                >
            > callAssignChargeToCell;
            callAssignChargeToCell(acc, forward(cursor), pos, flux[dir]);
        }

    };

    /** add current of a moving particle to the global current
     *
     * @param dataBoxJ DataBox with current field
     * @param pos1 current position of the particle
     * @param velocity velocity of the macro particle
     * @param charge charge of the macro particle
     * @param deltaTime dime difference of one simulation time step
     */
    template<typename DataBoxJ, typename PosType, typename VelType, typename ChargeType, typename T_Acc >
    DINLINE void operator()(const T_Acc& acc,
                            DataBoxJ dataBoxJ,
                            const PosType pos1,
                            const VelType velocity,
                            const ChargeType charge, const float_X deltaTime)
    {

        floatD_X deltaPos;
        for (uint32_t d = 0; d < simDim; ++d)
            deltaPos[d] = (velocity[d] * deltaTime) / cellSize[d];

        /*note: all positions are normalized to the grid*/
        floatD_X pos[2];
        pos[0] = (pos1 - deltaPos);
        pos[1] = (pos1);

        DataSpace<simDim> I[2];
        floatD_X relayPoint;


        for (int l = 0; l < 2; ++l)
        {
            for (uint32_t d = 0; d < simDim; ++d)
            {
                I[l][d] = math::floor(pos[l][d]);
            }
        }
        for (uint32_t d = 0; d < simDim; ++d)
        {
            relayPoint[d] = calc_relayPoint(I[0][d], I[1][d], pos[0][d], pos[1][d]);
        }
        const float_X volume_reci = float_X(1.0) / float_X(CELL_VOLUME);


        /* We have to use float as loop variable due to an nvcc bug
         * If we use int than `float_X sign = float_X(1.) - float_X(2.) * l;`
         * creates wrong results
         * it can be the same bug as
         * @see https://devtalk.nvidia.com/default/topic/752200/cuda-programming-and-performance/nvcc-loop-bug-since-cuda-5-5/
         */
        for (float_32 l = 0; l < 2; ++l)
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
                const float_X tmpRelayPoint = relayPoint[d];
                inCellPos[d] = calc_InCellPos(pos_tmp, tmpRelayPoint, I[parId][d]);
                /* We multiply with `cellSize[d]` due to the fact that the attribute for the
                 * in-cell particle `position` (and it's change in DELTA_T) is normalize to [0,1) */
                flux[d] = sign * calc_chargeFlux(pos_tmp, tmpRelayPoint, deltaTime, charge) * volume_reci * cellSize[d];
            }

            /* this loop is only needed for 2D, we need a flux in z direction */
            for (uint32_t d = simDim; d < 3; ++d)
            {
                /* in 2D the full flux for the z direction is given to the virtual particle zero */
                flux[d] = (parId == 0 ? charge * velocity[d] * volume_reci : float_X(0.0));
            }

            auto cursorJ = dataBoxJ.shift(precisionCast<int>(I[parId])).toCursor();

            // the current has three components
            using ComponentsRange = boost::mpl::range_c< int, 0, 3 >;
            // this transformation is needed to use boost::ml::accumulate on ComponentsRange
            using Components = typename MakeSeq< ComponentsRange >::type;

            // calculate x,y,z component of the current
            ForEach<
                Components,
                AssignOneDirection< bmpl::_1 >
            > callAssignOneDirection;
            callAssignOneDirection(acc, forward(cursorJ), inCellPos, flux);
        }
    }

    static pmacc::traits::StringProperty getStringProperties()
    {
        pmacc::traits::StringProperty propList( "name", "ZigZag" );
        return propList;
    }

private:

    /** calculate virtual point were we split our particle trajectory
     *
     * The relay point calculation differs from the paper version in the point
     * that the trajectory of a particle which does not leave the cell is not splitted.
     * The relay point for a particle which does not leave the cell is set to the
     * current position `x_2`
     *
     * @param i_1 grid point which is less than x_1 (`i_1=floor(x_1)`)
     * @param i_2 grid point which is less than x_2 (`i_2=floor(x_2)`)
     * @param x_1 begin position of the particle trajectory
     * @param x_2 end position of the particle trajectory
     * @return relay point for particle trajectory
     */
    DINLINE float_X
    calc_relayPoint(const float_X i_1, const float_X i_2, const float_X x_1, const float_X x_2) const
    {
        /* paper version:
         *   i_1 == i_2 ? (x_1 + x_2) / float_X(2.0) : math::max(i_1, i_2);
         */
        return i_1 == i_2 ? x_2 : math::max(i_1, i_2);
    }

    /** get normalized average in cell particle position
     *
     * @param x position of the particle begin of the trajectory
     * @param x_r position of the particle end of the trajectory
     * @param i grid point which is less than x (`i=floor(x)`)
     * @return average in cell position
     */
    DINLINE float_X
    calc_InCellPos(const float_X x, const float_X x_r, const float_X i) const
    {
        return (x + x_r) / (float_X(2.0)) - i;
    }

    /** get charge flux
     *
     * for F_2 call with -x and -x_r
     * or
     * negate the result of this method (e.g. `-calc_F(...)`)
     *
     * @param x position of the particle begin of the trajectory
     * @param x_r position of the particle end of the trajectory
     * @param delta_t time difference of one simulation time step
     * @param q charge of the particle
     * @return flux of the moving particle
     */
    DINLINE float_X
    calc_chargeFlux(const float_X x, const float_X x_r, const float_X delta_t, const float_X q) const
    {
        return q * (x_r - x) / delta_t;
    }
};

} //namespace currentSolver

} //namespace picongpu
