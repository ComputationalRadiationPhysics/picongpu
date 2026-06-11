/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

// need dimensions.param
#include "picongpu/defines.hpp"

namespace picongpu::particles::atomicPhysics
{
    //! short hand methods for getting dataBox access indices in atomicPhysics from kernels
    struct KernelIndexation
    {
        /** get index of superCell corresponding of the worker
         *
         * @attention assumes that the kernel was launched for CORE+BORDER Region
         */
        template<typename T_Worker, typename T_AreaMapping>
        HDINLINE static pmacc::DataSpace<picongpu::simDim> getSuperCellIndex(
            T_Worker const& worker,
            T_AreaMapping const areaMapping)
        {
            static_assert(T_AreaMapping::AreaType == CORE + BORDER, "kernel area needs to be CORE+BORDER");

            return areaMapping.getSuperCellIndex(worker.blockDomIdxND());
        }

        /** get index of SuperCellField entry corresponding to the worker
         *
         * @attention assumes that the kernel was launched for CORE+BORDER Region
         */
        template<typename T_Worker, typename T_AreaMapping>
        HDINLINE static pmacc::DataSpace<picongpu::simDim> getSuperCellFieldIndex(
            T_Worker const& worker,
            T_AreaMapping const areaMapping)
        {
            // atomicPhysics superCellFields have no guard, but areaMapping includes a guard
            //  -> must subtract guard to get correct superCellFieldIdx
            return getSuperCellIndex(worker, areaMapping) - areaMapping.getGuardingSuperCells();
        }

        /** get index of SuperCellField entry corresponding to the worker
         *
         * @details version for already known superCellIndex
         * @attention assumes that the kernel was launched for CORE+BORDER Region
         */
        template<typename T_AreaMapping>
        HDINLINE static pmacc::DataSpace<picongpu::simDim> getSuperCellFieldIndexFromSuperCellIndex(
            T_AreaMapping const areaMapping,
            pmacc::DataSpace<picongpu::simDim> const superCellIndex)
        {
            static_assert(T_AreaMapping::AreaType == CORE + BORDER, "kernel area needs to be CORE+BORDER");

            // atomicPhysics superCellFields have no guard, but areMapping includes a guard
            //  -> must subtract guard to get correct superCellFieldIdx
            return superCellIndex - areaMapping.getGuardingSuperCells();
        }
    };
} // namespace picongpu::particles::atomicPhysics
