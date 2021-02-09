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

                class LocalRateMatrix : public ISimulationData
                {
                private:
                    /** A[iz, numpop, numpop] */
                    using RateMatrix = memory::Array<
                        memory::Array<
                            memory::Array<float_X, picongpu::flylite::populations>,
                            picongpu::flylite::populations>,
                        picongpu::flylite::ionizationStates>;
                    GridBuffer<RateMatrix, simDim>* m_rateMatrix;
                    std::string m_speciesName;

                public:
                    /** Allocate and initialize local rate matrix for ion state transitions
                     *
                     * @param histSizeLocal spatial size of the local energy histogram
                     */
                    LocalRateMatrix(std::string const& ionSpeciesName, DataSpace<simDim> const& histSizeLocal)
                        : m_rateMatrix(nullptr)
                        , m_speciesName(ionSpeciesName)
                    {
                        m_rateMatrix = new GridBuffer<RateMatrix, simDim>(histSizeLocal);
                    }

                    ~LocalRateMatrix()
                    {
                        __delete(m_rateMatrix);
                    }

                    static std::string getName(std::string const& speciesGroup)
                    {
                        return speciesGroup + "_RateMatrix";
                    }

                    std::string getName()
                    {
                        return getName(m_speciesName);
                    }

                    GridBuffer<RateMatrix, simDim>& getGridBuffer()
                    {
                        return *m_rateMatrix;
                    }

                    /* implement ISimulationData members */
                    void synchronize() override
                    {
                        m_rateMatrix->deviceToHost();
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
