/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics::debug
{
    /** debug only, write atomicPhysics attributes to console
     *
     * @attention only creates output if compiling for debug backend
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
            std::cout << "\t - mask: " << ((ion[multiMask_]) ? "true" : "false") << std::endl;

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
