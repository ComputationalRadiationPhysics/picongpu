/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file collection of helper functions for passing with ionization potential depression inputs to the IPDModell call

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    struct PassIPDInputs
    {
        template<typename T_IPDModel, typename T_RNGFactory, typename T_ChargeStateDataBox, typename... T_IPDInput>
        HDINLINE static typename T_IPDModel::SuperCellConstantInput getSuperCellConstantInput_RNGFactory(
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx,
            T_RNGFactory&,
            T_ChargeStateDataBox,
            T_IPDInput... ipdInput)
        {
            return T_IPDModel::getSuperCellConstantInput(superCellFieldIdx, ipdInput...);
        }

        template<typename T_IPDModel, typename T_ChargeStateDataBox, typename... T_IPDInput>
        HDINLINE static typename T_IPDModel::SuperCellConstantInput getSuperCellConstantInput(
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx,
            T_ChargeStateDataBox,
            T_IPDInput... ipdInput)
        {
            return T_IPDModel::getSuperCellConstantInput(superCellFieldIdx, ipdInput...);
        }

        template<typename T_RNGFactory, typename... T_AddStuff>
        HDINLINE static T_RNGFactory& extractRNGFactory(T_RNGFactory& rngFactory, T_AddStuff...)
        {
            return rngFactory;
        }

        template<typename T_RNGFactory, typename T_ChargeStateDataBox, typename... T_IPDInput>
        HDINLINE static T_ChargeStateDataBox extractChargeStateBox_RngFactory(
            T_RNGFactory&,
            T_ChargeStateDataBox chargeStateBox,
            T_IPDInput...)
        {
            return chargeStateBox;
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
