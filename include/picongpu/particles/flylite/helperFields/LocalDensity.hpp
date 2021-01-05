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
#include <memory>


namespace picongpu
{
    namespace particles
    {
        namespace flylite
        {
            namespace helperFields
            {
                class LocalDensity : public ISimulationData
                {
                public:
                    using ValueType = float_X;

                private:
                    GridBuffer<ValueType, simDim>* m_density;
                    std::string m_speciesGroup;

                public:
                    /** Allocate and initialize local (number) density
                     *
                     * @param speciesGroup unique naming for the species inside this density,
                     *                     e.g. a collection of electron species or ions
                     * @param sizeLocal spatial size of the local density value
                     */
                    LocalDensity(std::string const& speciesGroup, DataSpace<simDim> const& sizeLocal)
                        : m_density(nullptr)
                        , m_speciesGroup(speciesGroup)
                    {
                        // without guards
                        m_density = new GridBuffer<ValueType, simDim>(sizeLocal);
                    }

                    ~LocalDensity()
                    {
                        __delete(m_density);
                    }

                    static std::string getName(std::string const& speciesGroup)
                    {
                        return speciesGroup + "_LocalDensity";
                    }

                    std::string getName()
                    {
                        return getName(m_speciesGroup);
                    }

                    GridBuffer<ValueType, simDim>& getGridBuffer()
                    {
                        return *m_density;
                    }

                    /* implement ISimulationData members */
                    void synchronize() override
                    {
                        m_density->deviceToHost();
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
