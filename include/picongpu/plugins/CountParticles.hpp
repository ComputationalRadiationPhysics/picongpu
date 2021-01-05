/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Rene Widera, Richard Pausch
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

#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/particles/filter/filter.hpp"

#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/nvidia/functors/Max.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <pmacc/particles/operations/CountParticles.hpp>

#include "common/txtFileHandling.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>


namespace picongpu
{
    using namespace pmacc;

    template<class ParticlesType>
    class CountParticles : public ISimulationPlugin
    {
    private:
        typedef MappingDesc::SuperCellSize SuperCellSize;

        MappingDesc* cellDescription;
        std::string notifyPeriod;

        std::string pluginName;
        std::string pluginPrefix;
        std::string filename;

        std::ofstream outFile;
        /*only rank 0 create a file*/
        bool writeToFile;

        mpi::MPIReduce reduce;

    public:
        CountParticles()
            : pluginName("CountParticles: count macro particles of a species")
            , pluginPrefix(ParticlesType::FrameType::getName() + std::string("_macroParticlesCount"))
            , filename(pluginPrefix + ".dat")
            , cellDescription(nullptr)
            , writeToFile(false)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~CountParticles()
        {
        }

        void notify(uint32_t currentStep)
        {
            countParticles<CORE + BORDER>(currentStep);
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            desc.add_options()(
                (pluginPrefix + ".period").c_str(),
                po::value<std::string>(&notifyPeriod),
                "enable plugin [for each n-th step]");
        }

        std::string pluginGetName() const
        {
            return pluginName;
        }

        void setMappingDescription(MappingDesc* cellDescription)
        {
            this->cellDescription = cellDescription;
        }

    private:
        void pluginLoad()
        {
            if(!notifyPeriod.empty())
            {
                writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());

                if(writeToFile)
                {
                    outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << filename << "] for output, disable plugin output. "
                                  << std::endl;
                        writeToFile = false;
                    }
                    // create header of the file
                    outFile << "#step count"
                            << " \n";
                }

                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
            }
        }

        void pluginUnload()
        {
            if(!notifyPeriod.empty())
            {
                if(writeToFile)
                {
                    outFile.flush();
                    outFile << std::endl; // now all data are written to file
                    if(outFile.fail())
                        std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                    outFile.close();
                }
            }
        }

        void restart(uint32_t restartStep, const std::string restartDirectory)
        {
            if(!writeToFile)
                return;

            writeToFile = restoreTxtFile(outFile, filename, restartStep, restartDirectory);
        }

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
        {
            if(!writeToFile)
                return;

            checkpointTxtFile(outFile, filename, currentStep, checkpointDirectory);
        }

        template<uint32_t AREA>
        void countParticles(uint32_t currentStep)
        {
            uint64_cu size;

            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            const DataSpace<simDim> localSize(subGrid.getLocalDomain().size);

            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName(), true);

            // enforce that the filter interface is fulfilled
            particles::filter::IUnary<particles::filter::All> parFilter{currentStep};

            /*count local particles*/
            size = pmacc::CountParticles::countOnDevice<AREA>(
                *particles,
                *cellDescription,
                DataSpace<simDim>(),
                localSize,
                parFilter);
            dc.releaseData(ParticlesType::FrameType::getName());

            uint64_cu reducedValueMax;
            if(picLog::log_level & picLog::CRITICAL::lvl)
            {
                reduce(nvidia::functors::Max(), &reducedValueMax, &size, 1, mpi::reduceMethods::Reduce());
            }


            uint64_cu reducedValue;
            reduce(nvidia::functors::Add(), &reducedValue, &size, 1, mpi::reduceMethods::Reduce());

            if(writeToFile)
            {
                if(picLog::log_level & picLog::CRITICAL::lvl)
                {
                    log<picLog::CRITICAL>("maximum number of  particles on a GPU : %d\n") % reducedValueMax;
                }

                outFile << currentStep << " " << reducedValue << " " << std::scientific << (float_64) reducedValue
                        << std::endl;
            }
        }
    };

} /* namespace picongpu */
