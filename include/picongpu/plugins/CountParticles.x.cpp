/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Rene Widera, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "common/txtFileHandling.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/PluginRegistry.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/operations/CountParticles.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace picongpu
{
    using namespace pmacc;

    template<class ParticlesType>
    class CountParticles : public ISimulationPlugin
    {
    private:
        MappingDesc* cellDescription{nullptr};
        std::string notifyPeriod;

        std::string pluginName;
        std::string pluginPrefix;
        std::string filename;

        std::ofstream outFile;
        /*only rank 0 create a file*/
        bool writeToFile{false};

        mpi::MPIReduce reduce;

    public:
        CountParticles()
            : pluginName("CountParticles: count macro particles of a species")
            , pluginPrefix(ParticlesType::FrameType::getName() + std::string("_macroParticlesCount"))
            , filename(pluginPrefix + ".dat")

        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        ~CountParticles() override = default;

        void notify(uint32_t currentStep) override
        {
            countParticles<CORE + BORDER>(currentStep);
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()(
                (pluginPrefix + ".period").c_str(),
                po::value<std::string>(&notifyPeriod),
                "enable plugin [for each n-th step]");
        }

        std::string pluginGetName() const override
        {
            return pluginName;
        }

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            this->cellDescription = cellDescription;
        }

    private:
        void pluginLoad() override
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

        void pluginUnload() override
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

        void restart(uint32_t restartStep, std::string const restartDirectory) override
        {
            if(!writeToFile)
                return;

            writeToFile = restoreTxtFile(outFile, filename, restartStep, restartDirectory);
        }

        void checkpoint(uint32_t currentStep, std::string const checkpointDirectory) override
        {
            if(!writeToFile)
                return;

            checkpointTxtFile(outFile, filename, currentStep, checkpointDirectory);
        }

        template<uint32_t AREA>
        void countParticles(uint32_t currentStep)
        {
            uint64_cu size;

            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
            DataSpace<simDim> const localSize(subGrid.getLocalDomain().size);

            DataConnector& dc = Environment<>::get().DataConnector();
            auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

            auto idProvider = dc.get<IdProvider>("globalId");
            // enforce that the filter interface is fulfilled
            particles::filter::IUnary<particles::filter::All> parFilter{currentStep, idProvider->getDeviceGenerator()};

            /*count local particles*/
            size = pmacc::CountParticles::countOnDevice<AREA>(
                *particles,
                *cellDescription,
                DataSpace<simDim>(),
                localSize,
                parFilter);

            uint64_cu reducedValueMax;
            if(picLog::log_level & picLog::CRITICAL::lvl)
            {
                reduce(pmacc::math::operation::Max(), &reducedValueMax, &size, 1, mpi::reduceMethods::Reduce());
            }


            uint64_cu reducedValue;
            reduce(pmacc::math::operation::Add(), &reducedValue, &size, 1, mpi::reduceMethods::Reduce());

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

PIC_REGISTER_SPECIES_PLUGIN(picongpu::CountParticles<boost::mpl::_1>);
