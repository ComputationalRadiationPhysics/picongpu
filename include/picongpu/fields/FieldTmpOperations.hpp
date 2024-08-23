/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch, Benjamin Worpitz, Pawel Ordyna
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldTmp.hpp"
#include "picongpu/fields/FieldTmp.kernel"
#include "picongpu/fields/MaxwellSolver/Solvers.hpp"
#include "picongpu/param/fileOutput.param"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/traits/GetInterpolation.hpp"
#include "picongpu/traits/GetMargin.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/SuperCellDescription.hpp>
#include <pmacc/fields/operations/AddExchangeToBorder.hpp>
#include <pmacc/fields/operations/CopyGuardToExchange.hpp>
#include <pmacc/fields/tasks/FieldFactory.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <memory>
#include <string>

namespace picongpu
{
    /** Compute an attribute derived from species in an area
     *
     * @tparam AREA area to compute values in
     * @tparam T_Species particle species type
     * @tparam Filter particle filter used to filter contributing particles
     *         (default is all particles contribute)
     *
     * @param species particle species
     * @param currentStep index of time iteration
     */
    template<uint32_t AREA, class FrameSolver, typename Filter = particles::filter::All, class ParticlesClass>
    inline void computeFieldTmpValue(FieldTmp& fieldTmp, ParticlesClass& parClass, uint32_t const)
    {
        using BlockArea = SuperCellDescription<
            typename MappingDesc::SuperCellSize,
            typename FrameSolver::LowerMargin,
            typename FrameSolver::UpperMargin>;

        auto mapper = makeStrideAreaMapper<AREA, 3>(parClass.getCellDescription());
        typename ParticlesClass::ParticlesBoxType pBox = parClass.getDeviceParticlesBox();

        auto fieldTmpBox = fieldTmp.getDeviceDataBox();
        FrameSolver solver;
        using ParticleFilter = typename Filter ::template apply<ParticlesClass>::type;
        const uint32_t currentStep = Environment<>::get().SimulationDescription().getCurrentStep();

        DataConnector& dc = Environment<>::get().DataConnector();
        auto idProvider = dc.get<IdProvider>("globalId");

        auto iFilter = particles::filter::IUnary<ParticleFilter>{currentStep, idProvider->getDeviceGenerator()};

        do
        {
            PMACC_LOCKSTEP_KERNEL(KernelComputeSupercells<BlockArea>{})
                .config(mapper.getGridDim(), pBox)(fieldTmpBox, pBox, solver, iFilter, mapper);
        } while(mapper.next());
    }

    /** Modify this field by an another field
     *
     * @tparam AREA area where the values are modified
     * @tparam T_ModifyingOperation a binary operation defining the result of the modification as a function
     *  of two values. The 1st value is this field and the 2nd value is the modifying field.
     * @tparam T_ModifyingField type of the second field
     *
     * @param modifyingField the second field
     */
    template<uint32_t AREA, typename T_ModifyingOperation, typename T_ModifyingField, typename T_CellDesc>
    inline void modifyFieldTmpByField(
        FieldTmp& fieldTmp,
        T_CellDesc const& cellDescription,
        T_ModifyingField& modifyingField)
    {
        auto mapper = makeAreaMapper<AREA>(cellDescription);

        auto fieldTmpBox = fieldTmp.getDeviceDataBox();
        const auto modifyingBox = modifyingField.getGridBuffer().getDeviceBuffer().getDataBox();

        using Kernel = ModifyByFieldKernel<T_ModifyingOperation, MappingDesc::SuperCellSize>;
        PMACC_LOCKSTEP_KERNEL(Kernel{}).config(
            mapper.getGridDim(),
            SuperCellSize{})(mapper, fieldTmpBox, modifyingBox);
    }

} // namespace picongpu
