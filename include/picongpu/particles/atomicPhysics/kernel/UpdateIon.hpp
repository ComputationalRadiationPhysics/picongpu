/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
