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
#include "simulationControl/DomainInformation.hpp"
#include "simulationControl/VirtualWindow.hpp"

namespace picongpu
{
using namespace PMacc;

/**
 * Domain and window information on a specific selected area.
 *
 * For a detailed description of domains and windows, see the PIConGPU wiki page:
 * https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
 */
struct SelectionInformation
{
    /** sizes and offsets of actual simulation data */
    DomainInformation domains;

    /** information on the moving window */
    VirtualWindow movingWindow;

    /** selection on this GPU */
    Selection<simDim> localSelection;

    /** selection over all GPUs */
    Selection<simDim> globalSelection;

    /**
     * Offset difference between local moving window and local domain if
     * start of local moving window > start of local domain, 0 otherwise. */
    DataSpace<simDim> selectionOffset;

    /**
     * Return a string representation of all members
     *
     * @return string representation of all members
     */
    HINLINE const std::string toString(void) const
    {
        std::stringstream str;
        str << "[ totalDomain = " << domains.totalDomain.toString() <<
                " globalDomain = " << domains.globalDomain.toString() <<
                " localDomain = " << domains.localDomain.toString() <<
                " globalMovingWindow = " << movingWindow.globalDimensions.toString() <<
                " localMovingWindow = " << movingWindow.localDimensions.toString() <<
                " localSelection = " << localSelection.toString() <<
                " globalSelection = " << globalSelection.toString() << " ]";

        return str.str();
    }
};

} // namespace picongpu

