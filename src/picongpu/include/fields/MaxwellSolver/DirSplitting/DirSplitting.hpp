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
#include <cuSTL/container/HostBuffer.hpp>

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
private:
    template<typename CursorE, typename CursorB, typename GridSize>
    void propagate(CursorE cursorE, CursorB cursorB, GridSize gridSize) const
    {
        typedef SuperCellSize BlockDim;

        algorithm::kernel::ForeachBlock<BlockDim> foreach;
        foreach(zone::SphericZone<3>(PMacc::math::Size_t<3>(BlockDim::x::value, gridSize.y(), gridSize.z())),
                cursor::make_NestedCursor(cursorE),
                cursor::make_NestedCursor(cursorB),
                DirSplittingKernel<BlockDim>((int)gridSize.x()));
    }
public:
    DirSplitting(MappingDesc) {}

        
    void update_beforeCurrent(uint32_t currentStep) const
    {
        typedef SuperCellSize GuardDim;

        DataConnector &dc = Environment<>::get().DataConnector();

        FieldE& fieldE = dc.getData<FieldE > (FieldE::getName(), true);
        FieldB& fieldB = dc.getData<FieldB > (FieldB::getName(), true);

        BOOST_AUTO(fieldE_coreBorder,
            fieldE.getGridBuffer().getDeviceBuffer().
                   cartBuffer().view(GuardDim().toRT(),
                                     -GuardDim().toRT()));
        BOOST_AUTO(fieldB_coreBorder,
            fieldB.getGridBuffer().getDeviceBuffer().
            cartBuffer().view(GuardDim().toRT(),
                              -GuardDim().toRT()));

        using namespace cursor::tools;
        using namespace PMacc::math;

        PMacc::math::Size_t<3> gridSize = fieldE_coreBorder.size();

        propagate(fieldE_coreBorder.origin(),
                  fieldB_coreBorder.origin(),
                  fieldE_coreBorder.size());

        __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
        __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));

        typedef PMacc::math::CT::Int<1,2,0> Orientation_Y;
        propagate(twistVectorFieldAxes<Orientation_Y>(fieldE_coreBorder.origin()),
                  twistVectorFieldAxes<Orientation_Y>(fieldB_coreBorder.origin()),
                  twistComponents<Orientation_Y>(gridSize));

        __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
        __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));

        typedef PMacc::math::CT::Int<2,0,1> Orientation_Z;
        propagate(twistVectorFieldAxes<Orientation_Z>(fieldE_coreBorder.origin()),
                  twistVectorFieldAxes<Orientation_Z>(fieldB_coreBorder.origin()),
                  twistComponents<Orientation_Z>(gridSize));


        if (laserProfile::INIT_TIME > float_X(0.0))
            dc.getData<FieldE > (FieldE::getName(), true).laserManipulation(currentStep);
        
        __setTransactionEvent(fieldE_withGuard.asyncCommunication(__getTransactionEvent()));
        __setTransactionEvent(fieldB_withGuard.asyncCommunication(__getTransactionEvent()));
        /*
        if(currentStep % 50 == 0)
        {
            container::HostBuffer<float3_X, 3> floatE_host(fieldE_withGuard.getGridBuffer().getDeviceBuffer().cartBuffer().size());
            floatE_host = fieldE_withGuard.getGridBuffer().getDeviceBuffer().cartBuffer();
            
            BOOST_AUTO(fieldE_RightGuard,
                floatE_host.view(PMacc::math::Int<3>(0, -8, 0)));
                
            std::ostringstream filename;
            filename << "RightGuard_" << currentStep << ".dat";
            std::ofstream file(filename.str().c_str());
            file << fieldE_RightGuard;
        }*/

        __setTransactionEvent(fieldE.asyncCommunication(__getTransactionEvent()));
        __setTransactionEvent(fieldB.asyncCommunication(__getTransactionEvent()));
    }

    void update_afterCurrent(uint32_t) const
    {    
    }
};

} // dirSplitting

} // picongpu
