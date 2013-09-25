/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, Heiko Burau, René Widera
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



#ifndef SIMSTARTINITIALISER_HPP
#define	SIMSTARTINITIALISER_HPP

#include "simulation_types.hpp"
#include "dataManagement/AbstractInitialiser.hpp"
#include "dataManagement/DataConnector.hpp"

namespace picongpu
{

/**
 * Simulation startup initialiser.
 * 
 * Initialises a new simulation from default values.
 * DataConnector has to be used with a FIFO compliant IDataSorter.
 * 
 * @tparam EBuffer type for Electrons (see MySimulation)
 * @tparam IBuffer type for Ions (see MySimulation)
 */
template <class EBuffer, class IBuffer>
class SimStartInitialiser : public AbstractInitialiser
{
public:

    void init(uint32_t id, ISimulationData& data, uint32_t currentStep)
    {
        // add ids for other types if necessary
        // fields are initialised by their constructor
        switch (id)
        {
        case PAR_ELECTRONS:
            initElectrons(static_cast<EBuffer&> (data), currentStep);
            break;

        case PAR_IONS:
            initIons(static_cast<IBuffer&> (data), currentStep);
            break;
        }
    }

    virtual ~SimStartInitialiser()
    {

    }

private:

    void initElectrons(EBuffer& electrons, uint32_t currentStep)
    {

        electrons.initFill(currentStep);

        electrons.deviceSetDrift(currentStep);
        if (ELECTRON_TEMPERATURE > float_X(0.0))
            electrons.deviceAddTemperature(ELECTRON_TEMPERATURE);
    }

    void initIons(IBuffer& ions, uint32_t currentStep)
    {

        //copy electrons' values to ions
        EBuffer &e_buffer = DataConnector::getInstance().getData<EBuffer>(PAR_ELECTRONS);

        ions.deviceCloneFrom(e_buffer);

        // must be called to overwrite cloned electron momenta,
        // so even for gamma=1.0
        ions.deviceSetDrift(currentStep);
        if (ION_TEMPERATURE > float_X(0.0))
            ions.deviceAddTemperature(ION_TEMPERATURE);
    }
};
}

#endif	/* SIMSTARTINITIALISER_HPP */

