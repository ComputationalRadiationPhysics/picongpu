/* Copyright 2024 Brian Marre
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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics::debug
{
    /** debug only, write atomicPhysics attributes to console
     *
     * @attention only creates ouptut if atomicPhysics debug setting CPU_OUTPUT_ACTIVE == True
     * @attention only useful if compiling serial backend
     */
    struct PrintAtomicPhysicsIonToConsole
    {
        template<typename T_Acc, typename T_Ion>
        HDINLINE auto operator()(T_Acc const&, T_Ion const& ion) const
            -> std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            std::cout << "ID: " << ion[particleId_] << std::endl;
            std::cout << "\t - weighting: " << ion[weighting_] << std::endl;

            std::cout << "\t - momentum: (" << ion[momentum_].toString(",", "") << ")" << std::endl;
            std::cout << "\t - position: (" << ion[position_].toString(",", "") << ")" << std::endl;
            std::cout << "\t - atomicPhysicsData:" << std::endl;
            std::cout << "\t\t - atomicStateCollectionIndex: " << ion[atomicStateCollectionIndex_] << std::endl;
            std::cout << "\t\t - processClass: " << static_cast<uint16_t>(ion[processClass_]) << std::endl;
            std::cout << "\t\t - transitionIndex: " << ion[transitionIndex_] << std::endl;
            std::cout << "\t\t - binIndex: " << ion[binIndex_] << std::endl;
            std::cout << "\t\t - accepted: " << ((ion[accepted_]) ? "true" : "false") << std::endl;
            std::cout << "\t\t - boundElectrons: " << ion[boundElectrons_] << std::endl;
        }

        template<typename T_Acc, typename T_Ion>
        HDINLINE auto operator()(T_Acc const&, T_Ion const& ion) const
            -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };
} // namespace picongpu::particles::atomicPhysics::debug
