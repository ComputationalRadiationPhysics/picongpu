/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
    struct TimeStepField : public SuperCellField<float_X, T_MappingDescription, false /*no guards*/>
    {
        TimeStepField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription, false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "TimeStepField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
