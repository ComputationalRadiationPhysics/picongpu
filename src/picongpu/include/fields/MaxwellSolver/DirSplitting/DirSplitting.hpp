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

#include "DirSplitting.def"

#include "simulation_defines.hpp"

#include <fields/MaxwellSolver/DirSplitting/DirSplitting.kernel>
#include <math/vector/Int.hpp>
#include <dataManagement/DataConnector.hpp>
#include <fields/FieldB.hpp>
#include <fields/FieldE.hpp>
#include "math/Vector.hpp"
#include <cuSTL/algorithm/kernel/ForeachBlock.hpp>
#include <lambda/Expression.hpp>
#include <cuSTL/cursor/NestedCursor.hpp>
#include <math/vector/TwistComponents.hpp>
#include <math/vector/compile-time/TwistComponents.hpp>

namespace picongpu
{
namespace dirSplitting
{
using namespace PMacc;

/** Check Directional Splitting grid and time conditions
 *
 * This is a workaround that the condition check is only
 * triggered if the current used solver is `DirSplitting`
 */
template<typename T_UsedSolver, typename T_Dummy=void>
struct ConditionCheck
{
};

template<typename T_Dummy>
struct ConditionCheck<DirSplitting, T_Dummy>
{
    /* Directional Splitting conditions:
     *
     * using SI units to avoid round off errors
     */
    PMACC_CASSERT_MSG(DirectionSplitting_Set_dX_equal_dt_times_c____check_your_gridConfig_param_file,
                      (SI::SPEED_OF_LIGHT_SI * SI::DELTA_T_SI) == SI::CELL_WIDTH_SI);
    PMACC_CASSERT_MSG(DirectionSplitting_use_cubic_cells____check_your_gridConfig_param_file,
                      SI::CELL_HEIGHT_SI == SI::CELL_WIDTH_SI);
#if (SIMDIM == DIM3)
    PMACC_CASSERT_MSG(DirectionSplitting_use_cubic_cells____check_your_gridConfig_param_file,
                      SI::CELL_DEPTH_SI == SI::CELL_WIDTH_SI);
#endif
};

class DirSplitting : private ConditionCheck<fieldSolver::FieldSolver>
{
public:
    typedef typename PMacc::math::CT::make_Int<simDim, 0>::type CurrentLowerMargin;
    typedef typename PMacc::math::CT::make_Int<simDim, 1>::type CurrentUpperMargin;

private:
    template<typename T_NavigatorTwist, typename T_AccessorTwist,typename T_JInterPolationDir,typename CursorE, typename CursorB, typename CursorJ, typename GridSize>
    void propagate(CursorE cursorE, CursorB cursorB,CursorJ cursorJ, GridSize gridSize) const
    {
        using namespace cursor::tools;
        using namespace PMacc::math;

        typedef typename CT::shrinkTo<T_NavigatorTwist, simDim>::type SimDimNavigatorTwist;

        // the grid size after the permutation
        PMACC_AUTO(gridSizeTwisted, twistComponents<
            SimDimNavigatorTwist
        >(gridSize));

        /* twist components of the supercell */
        typedef typename CT::TwistComponents<SuperCellSize, SimDimNavigatorTwist>::type BlockDim;

        PMacc::math::Size_t<simDim> zoneSize(gridSizeTwisted);
        // the x dimension of the zone is not parallelized
        zoneSize.x()=BlockDim::x::value;

        algorithm::kernel::ForeachBlock<BlockDim> foreach;
        foreach(zone::SphericZone<simDim>(zoneSize),
                cursor::make_NestedCursor(twistVectorFieldAxes(cursorE, SimDimNavigatorTwist(), T_AccessorTwist())),
                cursor::make_NestedCursor(twistVectorFieldAxes(cursorB, SimDimNavigatorTwist(), T_AccessorTwist())),
                cursor::make_NestedCursor(twistVectorFieldAxes(cursorJ, SimDimNavigatorTwist(), T_AccessorTwist())),
                DirSplittingKernel<BlockDim, T_JInterPolationDir>((int)gridSizeTwisted.x()));
    }
public:
    DirSplitting(MappingDesc) {}

    void update_afterCurrent(uint32_t currentStep) const
    {
        typedef SuperCellSize GuardDim;

        DataConnector &dc = Environment<>::get().DataConnector();

        FieldE& fieldE = dc.getData<FieldE > (FieldE::getName(), true);
        FieldB& fieldB = dc.getData<FieldB > (FieldB::getName(), true);
        FieldJ& fieldJ = dc.getData<FieldJ > (FieldJ::getName(), true);

        using namespace cursor::tools;

        BOOST_AUTO(fieldE_coreBorder,
            fieldE.getGridBuffer().getDeviceBuffer().
                   cartBuffer().view(GuardDim().toRT(),
                                     -GuardDim().toRT()));
        BOOST_AUTO(fieldB_coreBorder,
            fieldB.getGridBuffer().getDeviceBuffer().
            cartBuffer().view(GuardDim().toRT(),
                              -GuardDim().toRT()));

        BOOST_AUTO(fieldJ_coreBorder,
            fieldJ.getGridBuffer().getDeviceBuffer().
            cartBuffer().view(GuardDim().toRT(),
                              -GuardDim().toRT()));

        if (laserProfile::INIT_TIME > float_X(0.0))
        {
            fieldE.laserManipulation(currentStep);
            // the laser manipulates the E field, therefor we need to communicate it
            __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
        }

        PMacc::math::Size_t<simDim> gridSize = fieldE_coreBorder.size();

        // propagation in X direction
        {
            // permutation vector for the current accessor
           typedef PMacc::math::CT::Int<0,1,2> AccessorTwist;
           // permutation vector for the navigator
           typedef PMacc::math::CT::Int<0,1,2> NavigatorTwist;

           // direction where the current component needs to be interpolated
           typedef PMacc::math::CT::Int<0,2,1> InterpolationDir;
           propagate<NavigatorTwist,AccessorTwist,InterpolationDir>(
                     fieldE_coreBorder.origin(),
                     fieldB_coreBorder.origin(),
                     fieldJ_coreBorder.origin(),
                     gridSize);

           __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
           __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));
        }

        // propagation in Y direction
        {
            // permutation vector for the current accessor
           typedef PMacc::math::CT::Int<1,2,0> AccessorTwist;
           // permutation vector for the navigator
           typedef PMacc::math::CT::Int<1,0,2> NavigatorTwist;

           // direction where the current component needs to be interpolated
           typedef PMacc::math::CT::Int<0,1,2> InterpolationDir;
           propagate<NavigatorTwist,AccessorTwist,InterpolationDir>(
                     fieldE_coreBorder.origin(),
                     fieldB_coreBorder.origin(),
                     fieldJ_coreBorder.origin(),
                     gridSize);

           __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
           __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));
        }

#if (SIMDIM==DIM3)
        // propagation in Z direction
        {
            // permutation vector for the current accessor
           typedef PMacc::math::CT::Int<2,0,1> AccessorTwist;
           // permutation vector for the navigator
           typedef PMacc::math::CT::Int<2,0,1> NavigatorTwist;

           // direction where the current component needs to be interpolated
           typedef PMacc::math::CT::Int<0,2,1> InterpolationDir;
           propagate<NavigatorTwist,AccessorTwist,InterpolationDir>(
                     fieldE_coreBorder.origin(),
                     fieldB_coreBorder.origin(),
                     fieldJ_coreBorder.origin(),
                     gridSize);

           __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
           __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));
        }
#endif
    }

    void update_beforeCurrent(uint32_t) const
    {
        // all calculations are done after the current is calculated
    }

    static PMacc::traits::StringProperty getStringProperties()
    {
        PMacc::traits::StringProperty propList( "name", "DS" );
        return propList;
    }
};

} // dirSplitting

} // picongpu
