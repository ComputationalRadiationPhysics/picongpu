/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file get last resort processClass from transition direction and active processes

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/enums/ChooseTransitionGroup.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::enums
{
    template<ChooseTransitionGroup T_ChooseTransitionGroup>
    struct LastResort;

    template<>
    struct LastResort<ChooseTransitionGroup::boundBoundUpward>
    {
        template<bool T_spontaneousDeexcitation>
        static constexpr uint8_t processClass()
        {
            return u8(ProcessClass::electronicExcitation);
        }
    };

    template<>
    struct LastResort<ChooseTransitionGroup::boundBoundDownward>
    {
        template<bool T_spontaneousDeexcitation>
        static constexpr uint8_t processClass()
        {
            if constexpr(T_spontaneousDeexcitation)
                return u8(ProcessClass::spontaneousDeexcitation);
            else
                return u8(ProcessClass::electronicDeexcitation);
        }
    };

    template<>
    struct LastResort<ChooseTransitionGroup::collisionalBoundFreeUpward>
    {
        static constexpr uint8_t processClass()
        {
            return u8(ProcessClass::electronicIonization);
        }
    };

    template<>
    struct LastResort<ChooseTransitionGroup::fieldBoundFreeUpward>
    {
        static constexpr uint8_t processClass()
        {
            return u8(ProcessClass::noChange);
        }
    };

    template<>
    struct LastResort<ChooseTransitionGroup::autonomousDownward>
    {
        static constexpr uint8_t processClass()
        {
            return u8(ProcessClass::autonomousIonization);
        }
    };
} // namespace picongpu::particles::atomicPhysics::enums
