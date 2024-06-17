/* Copyright 2022-2023 Brian Marre
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

/** @file binElectrons sub-stage of atomicPhysics
 *
 * implements the binning of all electron species into a super cell field shared histogram for use with the
 * atomicPhysics rate calculation.
 */

/** @todo Tests, Brian Marre, 2023: test need to check correct binning in following cases
 * 1. no temperature and no drift, single species, bin to bin-0
 *      else: incorrect binning
 * 2. 200keV drift, single species, bin to 200keV bin
 *      else: incorrect binning
 * ( temperature, single species, correct distribution )
 * 3. 2 step simulation, single species, periodic boundaries, no "doubling" of histogram bin values
 *      else: missing reset between steps
 * 4. no temperature and no drift, multiple species, all species weights are actually accounted
 *      else: spurious reset in between different species calls
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/kernel/BinElectrons.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::stage
{
    /** atomicPhysics sub-stage for a species calling the kernel per superCell
     *
     * is called once per time step for the entire local simulation volume and for
     * every isElectron species by the atomicPhysics stage by the atomicPhysicsStage
     *
     * @tparam T_ElectronSpecies species for which to call the functor
     */
    template<typename T_ElectronSpecies>
    struct BinElectrons
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ElectronSpecies
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localTimeRemainingField = *dc.get<
                picongpu::particles::atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                "LocalTimeRemainingField");

            // pointer to memory, we will only work on device, no sync required
            // init pointer to electrons and localElectronHistogramField
            auto& electrons = *dc.get<ElectronSpecies>(ElectronSpecies::FrameType::getName());
            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            using BinElectrons = picongpu::particles::atomicPhysics::kernel::BinElectronsKernel<
                picongpu::atomicPhysics::ElectronHistogram>;

            // macro for call of kernel on every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(BinElectrons())
                .config(mapper.getGridDim(), electrons)(
                    mapper,
                    localTimeRemainingField.getDeviceDataBox(),
                    electrons.getDeviceParticlesBox(),
                    localElectronHistogramField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics::stage
