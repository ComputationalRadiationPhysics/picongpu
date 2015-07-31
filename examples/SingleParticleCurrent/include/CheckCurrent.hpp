/**
 * Copyright 2013 Heiko Burau, Rene Widera
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

#include "fields/FieldJ.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/container/view/View.hpp"
#include "cuSTL/cursor/MultiIndexCursor.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "math/Vector.hpp"

namespace picongpu
{

struct CheckCurrent
{
    struct PrintNonZeroComponents
    {
        typedef void type;
        float3_X totalJ;

        PrintNonZeroComponents() : totalJ(float3_X::create(0.0)) {}
        ~PrintNonZeroComponents()
        {
            const float_X unit_current = UNIT_CHARGE / (UNIT_LENGTH * UNIT_LENGTH * UNIT_TIME);
            totalJ *= unit_current;
            printf("totalJ: (%g, %g, %g) A/m²\n", totalJ.x(), totalJ.y(), totalJ.z());
        }

        HDINLINE void operator()(float3_X data, PMacc::math::Int<3> cellIdx)
        {

            const float_X unit_current = UNIT_CHARGE / (UNIT_LENGTH * UNIT_LENGTH * UNIT_TIME);
            if(data.x() != 0.0f)
                printf("j_x = %g at %d, %d, %d\n", data.x() * unit_current, cellIdx.x(), cellIdx.y(), cellIdx.z());
            if(data.y() != 0.0f)
                printf("j_y = %g at %d, %d, %d\n", data.y() * unit_current, cellIdx.x(), cellIdx.y(), cellIdx.z());
            if(data.z() != 0.0f)
                printf("j_z = %g at %d, %d, %d\n", data.z() * unit_current, cellIdx.x(), cellIdx.y(), cellIdx.z());

            this->totalJ += data;
        }
    };
    void operator ()(FieldJ& _fieldJ_device)
    {

        typedef SuperCellSize GuardDim;

        // Get fieldJ without guards
        BOOST_AUTO(fieldJ_device,
            _fieldJ_device.getGridBuffer().getDeviceBuffer().cartBuffer());

        container::HostBuffer<float3_X, 3> fieldJ_with_guards(fieldJ_device.size());
        fieldJ_with_guards = fieldJ_device;
        container::View<container::HostBuffer<float3_X, 3> > fieldJ(fieldJ_with_guards.view(GuardDim::toRT(), -GuardDim::toRT()));

        float3_X beta(BETA0_X, BETA0_Y, BETA0_Z);

        std::cout << "\nsingle P A R T I C L E facts:\n\n";
        std::cout << "position: (" << float3_X(LOCAL_POS_X, LOCAL_POS_Y, LOCAL_POS_Z)
            << ") at cell " << fieldJ.size()/size_t(2) << std::endl;
        std::cout << "velocity: (" << beta << ") * c\n";
        std::cout << "delta_pos: (" << beta * SPEED_OF_LIGHT / float3_X(CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH) << ") * cellSize\n";

        const float_64 j = BASE_CHARGE / CELL_VOLUME * abs(beta) * SPEED_OF_LIGHT;
        const float_64 unit_current = UNIT_CHARGE / (UNIT_LENGTH * UNIT_LENGTH * UNIT_TIME);
        std::cout << "j = rho * abs(velocity) = " << std::setprecision(6) << j * unit_current << " A/m²" << std::endl;
        std::cout << "------------------------------------------\n\n";

        std::cout << "fieldJ facts:\n\n";
//        std::cout << "zone: " << fieldJ.zone().size << ", " << fieldJ.zone().offset << std::endl;
//        std::cout << "index: " << *cursor::make_MultiIndexCursor<3>()(math::Int<3>(1,2,3)) << std::endl;
//        std::cout << "index: " << cursor::make_MultiIndexCursor<3>()[math::Int<3>(1,2,3)] << std::endl;

        algorithm::host::Foreach()(
            fieldJ.zone(),
            fieldJ.origin(), cursor::make_MultiIndexCursor<3>(),
            PrintNonZeroComponents());

        std::cout << "------------------------------------------\n\n";
    }
};


}
