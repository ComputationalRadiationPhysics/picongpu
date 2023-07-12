/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it and or modify
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

//! @file collection of helper functions for passing with ionization potential depression inputs to the IPDModell call

#pragma once

namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
{
    struct PassIPDInputs
    {
        template<typename T_IPDModel, typename T_RNGFactory, typename T_ChargeStateDataBox, typename... T_IPDInput>
        HDINLINE static float_X calculateIPD_RngFactory(
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx,
            T_RNGFactory&,
            T_ChargeStateDataBox chargeStateBox,
            T_IPDInput... ipdInput)
        {
            return T_IPDModel::template calculateIPD<T_ChargeStateDataBox::atomicNumber>(
                superCellFieldIdx,
                ipdInput...);
        }

        template<typename T_IPDModel, typename T_ChargeStateDataBox, typename... T_IPDInput>
        HDINLINE static float_X calculateIPD(
            pmacc::DataSpace<picongpu::simDim> const superCellFieldIdx,
            T_ChargeStateDataBox chargeStateBox,
            T_IPDInput... ipdInput)
        {
            return T_IPDModel::template calculateIPD<T_ChargeStateDataBox::atomicNumber>(
                superCellFieldIdx,
                ipdInput...);
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
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
