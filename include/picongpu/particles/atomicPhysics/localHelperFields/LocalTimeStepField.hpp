/* Copyright 2022-2024 Brian Marre
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file implements the local timeStepField for each superCell
 *
 * timeStep length for the current atomicPhysics iteration in each superCell
 */

#pragma once

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    //! debug only, write timeStep to console
    struct PrintTimeStepToConsole
    {
        // cpu version
        template<typename T_Acc>
        HDINLINE auto operator()(T_Acc const&, float_X const timeStep, pmacc::DataSpace<picongpu::simDim> superCellIdx)
            const -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            printf("timeStep %s: %.8e\n", superCellIdx.toString(",", "[]").c_str(), timeStep);
        }

        // gpu version, does nothing
        template<typename T_Acc>
        HDINLINE auto operator()(T_Acc const&, float_X const timeStep, pmacc::DataSpace<picongpu::simDim> superCellIdx)
            const -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /**@class superCell field of the current timeStep:float_X for one atomicPhysics iteration
     *
     * unit: sim.unit.time()
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct LocalTimeStepField : public SuperCellField<float_X, T_MappingDescription, false /*no guards*/>
    {
        LocalTimeStepField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription, false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "LocalTimeStepField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
