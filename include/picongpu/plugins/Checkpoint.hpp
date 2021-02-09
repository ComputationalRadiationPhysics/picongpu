/* Copyright 2017-2021 Rene Widera, Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/output/IIOBackend.hpp"
#include "picongpu/plugins/multi/IHelp.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"

#if(ENABLE_ADIOS == 1)
#    include "picongpu/plugins/adios/ADIOSWriter.hpp"
#endif
#if(ENABLE_OPENPMD == 1)
#    include "picongpu/plugins/openPMD/openPMDWriter.hpp"
#endif
#include <pmacc/pluginSystem/PluginConnector.hpp>

#include <string>
#include <map>
#include <memory>
#include <stdexcept>


namespace picongpu
{
    /** Checkpoint creation and load
     *
     *  Plugin is handling different IO-backends to create and load simulation checkpoints
     */
    class Checkpoint : public ISimulationPlugin
    {
    public:
        Checkpoint() : checkpointFilename("checkpoint"), restartChunkSize(0u)
        {
#if(ENABLE_OPENPMD == 1)
            ioBackendsHelp["openPMD"] = std::shared_ptr<plugins::multi::IHelp>(openPMD::openPMDWriter::getHelp());
#endif
#if(ENABLE_ADIOS == 1)
            ioBackendsHelp["adios"] = std::shared_ptr<plugins::multi::IHelp>(adios::ADIOSWriter::getHelp());
#endif
            // if adios is enabled the default is adios
            if(!ioBackendsHelp.empty())
            {
                checkpointBackendName = ioBackendsHelp.begin()->first;
                restartBackendName = ioBackendsHelp.begin()->first;
            }

            uint32_t backendCount = 0u;
            for(auto& backend : ioBackendsHelp)
            {
                if(backendCount >= 1u)
                    activeBackends += ", ";
                activeBackends += backend.first;
                ++backendCount;
            }

            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~Checkpoint()
        {
        }

        void pluginRegisterHelp(boost::program_options::options_description& desc)
        {
            namespace po = boost::program_options;
            if(ioBackendsHelp.empty())
                desc.add_options()("checkpoint", "plugin disabled [compiled without dependency HDF5 or Adios]");
            else
                desc.add_options()(
                    "checkpoint.backend",
                    po::value<std::string>(&checkpointBackendName),
                    (std::string("Optional backend for checkpointing [") + activeBackends
                     + "] default: " + checkpointBackendName)
                        .c_str())(
                    "checkpoint.file",
                    po::value<std::string>(&checkpointFilename),
                    "Optional checkpoint filename (prefix)")(
                    "checkpoint.restart.backend",
                    po::value<std::string>(&restartBackendName),
                    (std::string("Optional backend for restarting [") + activeBackends
                     + "] default: " + restartBackendName)
                        .c_str())(
                    "checkpoint.restart.file",
                    po::value<std::string>(&restartFilename),
                    "checkpoint restart filename (prefix)")(
                    /* 1,000,000 particles are around 3900 frames at 256 particles per frame
                     * and match ~30MiB with typical picongpu particles.
                     * The only reason why we use 1M particles per chunk is that we can get a
                     * frame overflow in our memory manager if we process all particles in one kernel.
                     **/
                    "checkpoint.restart.chunkSize",
                    po::value<uint32_t>(&restartChunkSize)->default_value(1000000u),
                    "Number of particles processed in one kernel call during restart to prevent frame count blowup");

            for(auto& backend : ioBackendsHelp)
                backend.second->expandHelp(desc, "checkpoint.");
        }

        std::string pluginGetName() const
        {
            return "Checkpoint";
        }

        void notify(uint32_t)
        {
        }

        void setMappingDescription(MappingDesc* cellDescription)
        {
            m_cellDescription = cellDescription;
        }

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
        {
            auto cBackend = ioBackends.find(checkpointBackendName);
            if(cBackend != ioBackends.end())
            {
                cBackend->second->dumpCheckpoint(currentStep, checkpointDirectory, checkpointFilename);
            }
        }

        void restart(uint32_t restartStep, const std::string restartDirectory)
        {
            auto rBackend = ioBackends.find(restartBackendName);
            if(rBackend != ioBackends.end())
            {
                rBackend->second->doRestart(restartStep, restartDirectory, restartFilename, restartChunkSize);
            }
        }

    private:
        void pluginLoad()
        {
            for(auto& backendHelp : ioBackendsHelp)
            {
                if(backendHelp.second->getNumPlugins() > 0u)
                    backendHelp.second->validateOptions();

                size_t const numSlaves = backendHelp.second->getNumPlugins();
                if(numSlaves > 1u)
                    throw std::runtime_error(
                        pluginGetName() + ": is no a multi plugin, each option can only be selected once.");
            }

            // create checkpoint creation backend
            if(!ioBackendsHelp.empty())
            {
                auto cBackendHelp = ioBackendsHelp.find(checkpointBackendName);
                if(cBackendHelp == ioBackendsHelp.end())
                    throw std::runtime_error(
                        std::string("IO-backend ") + checkpointBackendName
                        + " for checkpoints not found, possible backends: " + activeBackends);
                else
                    ioBackends[checkpointBackendName] = std::static_pointer_cast<IIOBackend>(
                        cBackendHelp->second->create(cBackendHelp->second, 0, m_cellDescription));
            }
            // create restart backend
            if(!ioBackendsHelp.empty() && checkpointBackendName != restartBackendName)
            {
                auto rBackend = ioBackendsHelp.find(restartBackendName);
                if(rBackend == ioBackendsHelp.end())
                    throw std::runtime_error(
                        std::string("IO-backend ") + restartBackendName
                        + " for restarts not found, possible backends: " + activeBackends);
                else
                    ioBackends[restartBackendName] = std::static_pointer_cast<IIOBackend>(
                        rBackend->second->create(rBackend->second, 0, m_cellDescription));
            }

            if(restartFilename.empty())
            {
                restartFilename = checkpointFilename;
            }
        }

        virtual void pluginUnload()
        {
            ioBackends.clear();
        }

        //! string list with all possible IO-backends
        std::string activeBackends;

        //! name of the IO-backend to create checkpoints
        std::string checkpointBackendName;
        //! prefix of the checkpoint file
        std::string checkpointFilename;

        //! name of the IO-backend to restart from a checkpoint
        std::string restartBackendName;
        //! prefix of the restart file
        std::string restartFilename;
        /** number of particles processed in one kernel
         *
         * Load particles in chunks avoid that the accelerator backend is
         * running out of frame memory.
         */
        uint32_t restartChunkSize;

        // can be "adios", "hdf5" and "openPMD"
        std::map<std::string, std::shared_ptr<IIOBackend>> ioBackends;

        std::map<std::string, std::shared_ptr<plugins::multi::IHelp>> ioBackendsHelp;

        MappingDesc* m_cellDescription = nullptr;
    };

} // namespace picongpu
