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

//! @file interface for all ionization potential depression(ipd) implementations

#pragma once

#include "picongpu/simulation_defines.hpp"


namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
{
    struct IPDModel
    {
        //! create all HelperFields required by the IPD model
        HINLINE static void createHelperFields();

        /** calculate all inputs for the ionization potential depression
         *
         * @tparam T_IPDIonSpeciesList list of all species partaking as ions in IPD input
         * @tparam T_IPDElectronSpeciesList list of all species partaking as electrons in IPD input
         *
         * @attention collective over all IPD species
         */
        template<typename T_IPDIonSpeciesList, typename T_IPDElectronSpeciesList>
        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep);

        /** check for and apply single step of pressure ionization cascade
         *
         * @attention assumes that ipd-input fields are up to date
         * @attention invalidates ipd-input fields if at least one ionization electron has been spawned
         *
         * @attention must be called once for each step in a pressure ionization cascade
         *
         * @tparam list of all species partaking as ion in atomicPhysics
         *
         * @attention collective over all ion species
         */
        template<typename T_AtomicPhysicsIonSpeciesList>
        HINLINE static void applyPressureIonization(
            picongpu::MappingDesc const mappingDesc,
            uint32_t const currentStep);

        /** calculate ionization potential depression
         *
         * @param superCellFieldIdx index of superCell in superCellField(without guards)
         * @param input to ipd calculation
         *
         * @return unit: eV, not weighted
         */
        template<typename... T_Input>
        HDINLINE static float_X calculateIPD(pmacc::DataSpace<simDim> const superCellFieldIdx, T_Input const... input);

        /** append ipd inut to kernelInput and do a PMACC_LOCKSTEP_KERNEL call for T_kernel
         *
         * @tparam T_Kernel kernel to call
         * @param kernelInput stuff to pass to the kernel, before the ionization potential depression input
         */
        template<typename T_Kernel, typename T_WorkerCfg, typename... T_KernelInput>
        HINLINE static void callKernelWithIPDInput(
            pmacc::DataConnector& dc,
            T_WorkerCfg workerCfg,
            pmacc::AreaMapping<CORE + BORDER, picongpu::MappingDesc>& mapper,
            T_KernelInput... kernelInput);
    };
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
