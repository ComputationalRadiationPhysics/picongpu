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

//! @file no ionization potential depression implementation

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/IPDInterface.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    struct NoIPD : IPDModel
    {
        //! create all HelperFields required by the IPD model
        HINLINE static void createHelperFields()
        {
        }

        template<uint32_t T_numberAtomicPhysicsIonSpecies>
        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc)
        {
        }

        //! no IPD, means no pressure ionization
        template<typename T_AtomicPhysicsIonSpeciesList>
        HINLINE static void applyIPDIonization(picongpu::MappingDesc const, uint32_t const)
        {
        }

        //! @returns 0._X eV
        HDINLINE static float_X calculateIPD()
        {
            return 0._X;
        }


        //! no input required, therefore straight pass through
        template<typename T_Kernel, uint32_t T_chunkSize, typename... T_KernelInput>
        HINLINE static void callKernelWithIPDInput(
            pmacc::DataConnector& dc,
            pmacc::AreaMapping<CORE + BORDER, picongpu::MappingDesc>& mapper,
            T_KernelInput... kernelInput)
        {
            PMACC_LOCKSTEP_KERNEL(T_Kernel())
                .template config<T_chunkSize>(mapper.getGridDim())(mapper, kernelInput...);
        }
    }
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
