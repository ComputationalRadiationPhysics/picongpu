/**
 * Copyright 2014 Felix Schmitt
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "simulationControl/Selection.hpp"
#include "mappings/simulation/SubGrid.hpp"

namespace picongpu
{
using namespace PMacc;

/**
 * Groups local, global and total domain information.
 *
 * For a detailed description of domains, see the PIConGPU wiki page:
 * https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
 */
struct DomainInformation
{
    DomainInformation()
    {
        const SimulationBox<simDim> &simBox = Environment<simDim>::get().SubGrid().getSimulationBox();

        totalDomain = Selection<simDim>(simBox.getGlobalSize());
        globalDomain = Selection<simDim>(simBox.getGlobalSize());
        localDomain = Selection<simDim>(simBox.getLocalSize(), simBox.getGlobalOffset());
    }

    /** total simulation volume, including active and inactive subvolumes */
    Selection<simDim> totalDomain;

    /** currently simulated volume over all GPUs, offset relative to totalDomain */
    Selection<simDim> globalDomain;

    /** currently simulated volume on this GPU, offset relative to globalDomain */
    Selection<simDim> localDomain;
};

} // namespace picongpu
