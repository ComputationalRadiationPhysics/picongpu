/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file implements the local timeRemainingField for each superCell
 *
 * timeRemaining for the current atomicPhysics step in each superCell
 */

#pragma once

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /** debug only, write timeRemaining to console
     *
     * @attention only creates ouptut if atomicPhysics debug setting CPU_OUTPUT_ACTIVE == True
     * @attention only useful if compiling for serial or cpu backend, otherwise will throw compile error if called by
     *  DumpSuperCellDataToConsole kernel on device
     */
    struct PrintTimeRemaingToConsole
    {
        //! cpu version
        template<typename T_Acc>
        HDINLINE auto operator()(
            T_Acc const&,
            float_X const timeRemaining,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            printf("timeRemaining %s: %.8e\n", superCellIdx.toString(",", "[]").c_str(), timeRemaining);
        }

        //! gpu version does nothing
        template<typename T_Acc>
        HDINLINE auto operator()(
            T_Acc const&,
            float_X const timeRemaining,
            pmacc::DataSpace<picongpu::simDim> superCellIdx) const
            -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /** holds a gridBuffer of the per-superCell timeRemaining:float_X for atomicPhysics
     *
     * unit: sim.unit.time()
     */
    template<typename T_MappingDescription>
    struct TimeRemainingField : public SuperCellField<float_X, T_MappingDescription, false /*no guards*/>
    {
        TimeRemainingField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription, false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "TimeRemainingField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
