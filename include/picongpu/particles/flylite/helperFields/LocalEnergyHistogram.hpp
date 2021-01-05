/* Copyright 2017-2021 Axel Huebl
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

// pmacc
#include <pmacc/dataManagement/ISimulationData.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/memory/Array.hpp>

#include <string>


namespace picongpu
{
    namespace particles
    {
        namespace flylite
        {
            namespace helperFields
            {
                using namespace pmacc;

                class LocalEnergyHistogram : public ISimulationData
                {
                private:
                    using EnergyHistogram = memory::Array<float_X, picongpu::flylite::energies>;
                    GridBuffer<EnergyHistogram, simDim>* m_energyHistogram;
                    std::string m_speciesGroup;

                public:
                    /** Allocate and Initialize local Energy Histogram
                     *
                     * @param speciesGroup unique naming for the species inside this histogram,
                     *                     e.g. a collection of electron species or photon species
                     * @param histSizeLocal spatial size of the local energy histogram
                     */
                    LocalEnergyHistogram(std::string const& speciesGroup, DataSpace<simDim> const& histSizeLocal)
                        : m_energyHistogram(nullptr)
                        , m_speciesGroup(speciesGroup)
                    {
                        m_energyHistogram = new GridBuffer<EnergyHistogram, simDim>(histSizeLocal);
                    }

                    ~LocalEnergyHistogram()
                    {
                        __delete(m_energyHistogram);
                    }

                    static std::string getName(std::string const& speciesGroup)
                    {
                        return speciesGroup + "_LocalEnergyHistogram";
                    }

                    std::string getName()
                    {
                        return getName(m_speciesGroup);
                    }

                    GridBuffer<EnergyHistogram, simDim>& getGridBuffer()
                    {
                        return *m_energyHistogram;
                    }

                    /* implement ISimulationData members */
                    void synchronize() override
                    {
                        m_energyHistogram->deviceToHost();
                    }

                    SimulationDataId getUniqueId() override
                    {
                        return getName();
                    }
                };

            } // namespace helperFields
        } // namespace flylite
    } // namespace particles
} // namespace picongpu
