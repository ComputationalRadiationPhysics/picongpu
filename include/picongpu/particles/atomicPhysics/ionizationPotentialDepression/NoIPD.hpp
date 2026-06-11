/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! @file no ionization potential depression implementation

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/IPDModel.hpp"

#include <pmacc/meta/ForEach.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    namespace detail
    {
        // the NoIPD model has no input, therefore the struct has no members
        struct NoIPDSuperCellConstantInput
        {
        };
    } // namespace detail

    struct NoIPD : IPDModel
    {
        using SuperCellConstantInput = detail::NoIPDSuperCellConstantInput;

        //! create all HelperFields required by the IPD model
        HINLINE static void createHelperFields(picongpu::DataConnector&, picongpu::MappingDesc const)
        {
        }

        template<
            uint32_t T_numberAtomicPhysicsIonSpecies,
            typename T_IPDIonSpeciesList,
            typename T_IPDElectronSpeciesList>
        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc, uint32_t const)
        {
        }

        //! no IPD, means no pressure ionization
        template<typename T_AtomicPhysicsIonSpeciesList, bool T_SkipFinishedSuperCell>
        HINLINE static void applyIPDIonization(picongpu::MappingDesc const, uint32_t const)
        {
        }

        HDINLINE static SuperCellConstantInput getSuperCellConstantInput(pmacc::DataSpace<simDim> const)
        {
            return SuperCellConstantInput();
        }

        //! @returns 0._X eV
        HDINLINE static float_X ipd(SuperCellConstantInput const, uint8_t const)
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
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
