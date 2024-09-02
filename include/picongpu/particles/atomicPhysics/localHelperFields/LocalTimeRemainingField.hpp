/* Copyright 2022-2023 Brian Marre
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
    struct LocalTimeRemainingField : public SuperCellField<float_X, T_MappingDescription, false /*no guards*/>
    {
        LocalTimeRemainingField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription, false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "LocalTimeRemainingField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
