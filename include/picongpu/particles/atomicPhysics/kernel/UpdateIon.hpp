/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2023-2024 Brian Marre
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::kernel
{
    namespace s_enums = picongpu::particles::atomicPhysics::enums;

    //! set ion to no-change transition
    template<typename T_Ion>
    HDINLINE void setNoChangeTransition(T_Ion& ion)
    {
        ion[processClass_] = u8(s_enums::ProcessClass::noChange);
        // no need to set ion[transitionIndex_] since already uniquely known by processClass = noChange
        // no-change transitions are not bin based therefore we don't set a bin, old values are ignored
        ion[accepted_] = true;
    }

    //! update ion with selected transition data, non-collisional version
    template<typename T_Ion>
    HDINLINE void updateIon(T_Ion& ion, uint8_t const selectedProcessClass, uint32_t const selectedTransitionIndex)
    {
        ion[processClass_] = selectedProcessClass;
        ion[transitionIndex_] = selectedTransitionIndex;
        // no need to set bin index for non-collisional transitions
        ion[accepted_] = true;
    }

    //! update ion with selected transition data, collisional version
    template<typename T_Ion>
    HDINLINE static void updateIon(
        T_Ion& ion,
        uint8_t const selectedProcessClass,
        uint32_t const selectedTransitionIndex,
        uint32_t const selectedBinIndex)
    {
        updateIon(ion, selectedProcessClass, selectedTransitionIndex);
        ion[binIndex_] = selectedBinIndex;
    }
} // namespace picongpu::particles::atomicPhysics::kernel
