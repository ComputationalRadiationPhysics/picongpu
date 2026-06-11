/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file check if processClass is in processClassesGroup

#pragma once

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics/enums/ProcessClassGroup.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::enums
{
    //! general interface for checking for if a processClass belongs to a processClassGroup
    template<ProcessClassGroup group>
    struct IsProcess
    {
        static constexpr bool check(uint8_t const processClass);
    };

    /** processClasses which are based on bound-bound transition data sets,
     *  "picongpu/particles/atomicPhysics/atomicData/" */
    template<>
    struct IsProcess<ProcessClassGroup::boundBoundBased>
    {
        static constexpr bool check(uint8_t const processClass)
        {
            if((processClass == u8(ProcessClass::electronicExcitation))
               || (processClass == u8(ProcessClass::electronicDeexcitation))
               || (processClass == u8(ProcessClass::spontaneousDeexcitation)))
                return true;
            return false;
        }
    };

    /** processClasses which are based on upward bound-free transition data sets,
     *  "picongpu/particles/atomicPhysics/atomicData/" */
    template<>
    struct IsProcess<ProcessClassGroup::boundFreeBased>
    {
        static constexpr bool check(uint8_t const processClass)
        {
            ///@todo implement recombination, Brian Marre, 2023
            if((processClass == u8(ProcessClass::electronicIonization))
               || (processClass == u8(ProcessClass::fieldIonization)))
                return true;
            return false;
        }
    };

    /** processClasses which are based on autonomous transition data sets,
     *  "picongpu/particles/atomicPhysics/atomicData/" */
    template<>
    struct IsProcess<ProcessClassGroup::autonomousBased>
    {
        static constexpr bool check(uint8_t const processClass)
        {
            if(processClass == u8(ProcessClass::autonomousIonization))
                return true;
            return false;
        }
    };

    //! processClass which causes ionization
    template<>
    struct IsProcess<ProcessClassGroup::ionizing>
    {
        static constexpr bool check(uint8_t const processClass)
        {
            if((processClass == u8(ProcessClass::electronicIonization))
               || (processClass == u8(ProcessClass::autonomousIonization))
               || (processClass == u8(ProcessClass::fieldIonization)))
                return true;
            return false;
        }
    };

    //! processClass describing interaction with free electron
    template<>
    struct IsProcess<ProcessClassGroup::electronicCollisional>
    {
        static constexpr bool check(uint8_t const processClass)
        {
            if((processClass == u8(ProcessClass::electronicExcitation))
               || (processClass == u8(ProcessClass::electronicDeexcitation))
               || (processClass == u8(ProcessClass::electronicIonization)))
                return true;
            return false;
        }
    };

    //! processClass describing interaction with electric field
    template<>
    struct IsProcess<ProcessClassGroup::electricFieldBased>
    {
        static constexpr bool check(uint8_t const processClass)
        {
            if(processClass == u8(ProcessClass::fieldIonization))
                return true;
            return false;
        }
    };

    //! processClass describing physical transition with initial state being the lowerState of transition
    template<>
    struct IsProcess<ProcessClassGroup::upward>
    {
        static constexpr bool check(uint8_t const processClass)
        {
            if((processClass == u8(ProcessClass::electronicExcitation))
               || (processClass == u8(ProcessClass::electronicIonization))
               || (processClass == u8(ProcessClass::fieldIonization)))
                return true;
            return false;
        }
    };

    //! processClass describing physical transition with initial state being the upperState of transition
    template<>
    struct IsProcess<ProcessClassGroup::downward>
    {
        static constexpr bool check(uint8_t const processClass)
        {
            if((processClass == u8(ProcessClass::electronicDeexcitation))
               || (processClass == u8(ProcessClass::spontaneousDeexcitation))
               || (processClass == u8(ProcessClass::autonomousIonization)))
                return true;
            return false;
        }
    };

} // namespace picongpu::particles::atomicPhysics::enums
