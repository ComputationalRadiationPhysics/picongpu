/* Copyright 2013-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include "picongpu/simulation_defines.hpp"
#include <pmacc/dataManagement/AbstractInitialiser.hpp>
//#include <pmacc/dataManagement/DataConnector.hpp>

#include <pmacc/Environment.hpp>

namespace picongpu
{

/**
 * Simulation startup initialiser.
 *
 * Initialises a new simulation from default values.
 * DataConnector has to be used with a FIFO compliant IDataSorter.
 *
 */
class SimStartInitialiser : public AbstractInitialiser
{
public:

    void init(ISimulationData& data, uint32_t currentStep)
    {

    }

    virtual ~SimStartInitialiser()
    {

    }
};
}

