/* Copyright 2013-2021 Heiko Burau, Rene Widera, Richard Pausch, Axel Huebl
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
#include <fstream>
#include <cstdio>
#include <pmacc/memory/buffers/GridBuffer.hpp>


namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            namespace frequencies_from_list
            {
                class FreqFunctor
                {
                public:
                    typedef GridBuffer<float_X, DIM1>::DataBoxType DBoxType;

                    FreqFunctor(void)
                    {
                    }

                    template<typename T>
                    FreqFunctor(T frequencies_handed)
                    {
                        this->frequencies_dev = frequencies_handed->getDeviceBuffer().getDataBox();
                        this->frequencies_host = frequencies_handed->getHostBuffer().getDataBox();
                    }

                    DINLINE float_X operator()(const unsigned int ID)
                    {
                        return (ID < radiation_frequencies::N_omega) ? frequencies_dev[ID] : 0.0;
                    }

                    HINLINE float_X get(const unsigned int ID)
                    {
                        return (ID < radiation_frequencies::N_omega) ? frequencies_host[ID] : 0.0;
                    }

                private:
                    DBoxType frequencies_dev;
                    DBoxType frequencies_host;
                };


                class InitFreqFunctor
                {
                public:
                    InitFreqFunctor(void)
                    {
                    }

                    ~InitFreqFunctor(void)
                    {
                        __delete(frequencyBuffer);
                    }

                    typedef GridBuffer<picongpu::float_X, DIM1>::DataBoxType DBoxType;

                    HINLINE void Init(const std::string path)
                    {
                        frequencyBuffer = new GridBuffer<float_X, DIM1>(DataSpace<DIM1>(N_omega));


                        DBoxType frequencyDB = frequencyBuffer->getHostBuffer().getDataBox();

                        std::ifstream freqListFile(path.c_str());
                        unsigned int i;

                        printf("freq: %s\n", path.c_str());

                        if(!freqListFile)
                        {
                            throw std::runtime_error(
                                std::string("The radiation-frequency-file ") + path
                                + std::string(" could not be found.\n"));
                        }


                        for(i = 0; i < N_omega && !freqListFile.eof(); ++i)
                        {
                            freqListFile >> frequencyDB[i];
                            // verbose output of loaded frequencies if verbose level PHYSICS is set:
                            log<PIConGPUVerboseRadiation::PHYSICS>("freq: %1% \t %2%") % i % frequencyDB[i];
                            frequencyDB[i] *= UNIT_TIME;
                        }

                        if(i != N_omega)
                        {
                            throw std::runtime_error(std::string("The number of frequencies in the list and the "
                                                                 "number of frequencies in the parameters differ.\n"));
                        }

                        frequencyBuffer->hostToDevice();
                    }

                    FreqFunctor getFunctor(void)
                    {
                        return FreqFunctor(frequencyBuffer);
                    }

                private:
                    GridBuffer<float_X, DIM1>* frequencyBuffer;
                };


            } // namespace frequencies_from_list
        } // namespace radiation
    } // namespace plugins
} // namespace picongpu
