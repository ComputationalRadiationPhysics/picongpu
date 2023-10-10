/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz
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

#if !ENABLE_OPENPMD
#    error The activated radiation plugin (radiation.param) requires openPMD-api.
#endif

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#include "picongpu/plugins/common/openPMDVersion.def"
#include "picongpu/plugins/common/stringHelpers.hpp"
#include "picongpu/plugins/radiation/Radiation.kernel"
#include "picongpu/plugins/radiation/executeParticleFilter.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/filesystem.hpp>
#include <pmacc/lockstep/lockstep.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/traits/HasIdentifier.hpp>

#include <algorithm> // std::any
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace radiation
        {
            using namespace pmacc;

            namespace po = boost::program_options;

            namespace idLabels
            {
                enum meshRecordLabelsEnum
                {
                    Amplitude = 0,
                    Detector = 1,
                    Frequency = 2
                };
            } // end namespace idLabels


            ///////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////  Radiation Plugin Class  ////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////

            template<class ParticlesType>
            class Radiation : public ISimulationPlugin
            {
                using Amplitude = picongpu::plugins::radiation::Amplitude<>;

            private:
                typedef MappingDesc::SuperCellSize SuperCellSize;

                typedef PIConGPUVerboseRadiation radLog;

                /**
                 * Object that stores the complex radiated amplitude on host and device.
                 * Radiated amplitude is a function of theta (looking direction) and
                 * frequency. Layout of the radiation array is:
                 * [omega_1(theta_1),omega_2(theta_1),...,omega_N-omega(theta_1),
                 *   omega_1(theta_2),omega_2(theta_2),...,omega_N-omega(theta_N-theta)]
                 * The second dimension is used to store intermediate results if command
                 * line option numJobs is > 1.
                 */
                std::unique_ptr<GridBuffer<Amplitude, 2>> radiation;
                radiation_frequencies::InitFreqFunctor freqInit;
                radiation_frequencies::FreqFunctor freqFkt;

                MappingDesc* cellDescription = nullptr;
                std::string notifyPeriod;
                uint32_t dumpPeriod = 0;
                uint32_t radStart;
                uint32_t radEnd;

                std::string pluginName;
                std::string speciesName;
                std::string pluginPrefix;
                std::string filename_prefix;
                bool totalRad = false;
                bool lastRad = false;
                std::string folderLastRad;
                std::string folderTotalRad;
                bool radPerGPU = false;
                std::string folderRadPerGPU;
                DataSpace<simDim> lastGPUpos;
                int numJobs;

                /**
                 * Data structure for storage and summation of the intermediate values of
                 * the calculated Amplitude from every host for every direction and
                 * frequency.
                 */
                std::vector<Amplitude> timeSumArray;
                std::vector<Amplitude> tmp_result;
                std::vector<vector_64> detectorPositions;
                std::vector<float_64> detectorFrequencies;

                bool isMaster = false;

                uint32_t currentStep = 0;
                uint32_t lastStep = 0;

                std::string pathRestart;
                std::string meshesPathName;

                mpi::MPIReduce reduce;
                static const int numberMeshRecords = 3;

                std::optional<::openPMD::Series> m_series;
                std::string openPMDSuffix = "_%T." + openPMD::getDefaultExtension(openPMD::ExtensionPreference::ADIOS);
                std::string openPMDExtensionCheckpointing
                    = openPMD::getDefaultExtension(openPMD::ExtensionPreference::ADIOS);
                std::string openPMDConfig = "{}";
                std::string openPMDCheckpointConfig = "{}";
                bool writeDistributedAmplitudes = false;

            public:
                Radiation()
                    : pluginName("Radiation: calculate the radiation of a species")
                    , speciesName(ParticlesType::FrameType::getName())
                    , pluginPrefix(speciesName + std::string("_radiation"))
                    , filename_prefix(pluginPrefix)
                    , meshesPathName("DetectorMesh/")
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                /**
                 * This function represents what is actually calculated if the plugin
                 * is called. Here, one only sets the particles pointer to the data of
                 * the latest time step and calls the 'calculateRadiationParticles'
                 * function if for the actual time step radiation is to be calculated.
                 * @param currentStep
                 */
                void notify(uint32_t currentStep) override
                {
                    if(currentStep >= radStart)
                    {
                        // radEnd = 0 is default, calculates radiation until simulation
                        // end
                        if(currentStep <= radEnd || radEnd == 0)
                        {
                            log<radLog::SIMULATION_STATE>("Radiation (%1%): calculate time step %2% ") % speciesName
                                % currentStep;

                            /* CORE + BORDER is PIC black magic, currently not needed
                             *
                             */
                            calculateRadiationParticles<CORE + BORDER>(currentStep);

                            log<radLog::SIMULATION_STATE>("Radiation (%1%): finished time step %2% ") % speciesName
                                % currentStep;
                        }
                    }
                }

                void pluginRegisterHelp(po::options_description& desc) override
                {
                    desc.add_options()(
                        (pluginPrefix + ".period").c_str(),
                        po::value<std::string>(&notifyPeriod),
                        "enable plugin [for each n-th step]")(
                        (pluginPrefix + ".dump").c_str(),
                        po::value<uint32_t>(&dumpPeriod)->default_value(0),
                        "dump integrated radiation from last dumped step [for each n-th step] (0 = only print data at "
                        "end of simulation)")(
                        (pluginPrefix + ".lastRadiation").c_str(),
                        po::bool_switch(&lastRad),
                        "enable calculation of integrated radiation from last dumped step")(
                        (pluginPrefix + ".folderLastRad").c_str(),
                        po::value<std::string>(&folderLastRad)->default_value("lastRad"),
                        "folder in which the integrated radiation from last dumped step is written")(
                        (pluginPrefix + ".totalRadiation").c_str(),
                        po::bool_switch(&totalRad),
                        "enable calculation of integrated radiation from start of simulation")(
                        (pluginPrefix + ".folderTotalRad").c_str(),
                        po::value<std::string>(&folderTotalRad)->default_value("totalRad"),
                        "folder in which the integrated radiation from start of simulation is written")(
                        (pluginPrefix + ".start").c_str(),
                        po::value<uint32_t>(&radStart)->default_value(2),
                        "time index when radiation should start with calculation")(
                        (pluginPrefix + ".end").c_str(),
                        po::value<uint32_t>(&radEnd)->default_value(0),
                        "time index when radiation should end with calculation")(
                        (pluginPrefix + ".radPerGPU").c_str(),
                        po::bool_switch(&radPerGPU),
                        "enable radiation output from each GPU individually")(
                        (pluginPrefix + ".folderRadPerGPU").c_str(),
                        po::value<std::string>(&folderRadPerGPU)->default_value("radPerGPU"),
                        "folder in which the radiation of each GPU is written")(
                        (pluginPrefix + ".numJobs").c_str(),
                        po::value<int>(&numJobs)->default_value(2),
                        "Number of independent jobs used for the radiation calculation.")(
                        (pluginPrefix + ".openPMDSuffix").c_str(),
                        po::value<std::string>(&openPMDSuffix)
                            ->default_value(
                                "_%T_0_0_0." + openPMD::getDefaultExtension(openPMD::ExtensionPreference::HDF5)),
                        "Suffix for openPMD filename extension and iteration expansion pattern.")(
                        (pluginPrefix + ".openPMDCheckpointExtension").c_str(),
                        po::value<std::string>(&openPMDExtensionCheckpointing)
                            ->default_value(openPMD::getDefaultExtension(openPMD::ExtensionPreference::HDF5)),
                        "Filename extension for openPMD checkpoints.")(
                        (pluginPrefix + ".openPMDConfig").c_str(),
                        po::value<std::string>(&openPMDConfig)->default_value("{}"),
                        "JSON/TOML configuration for initializing openPMD.")(
                        (pluginPrefix + ".openPMDCheckpointConfig").c_str(),
                        po::value<std::string>(&openPMDCheckpointConfig)->default_value("{}"),
                        "JSON/TOML configuration for initializing openPMD checkpointing.")(
                        (pluginPrefix + ".distributedAmplitude").c_str(),
                        po::value<bool>(&writeDistributedAmplitudes)->default_value(false),
                        "Additionally output distributed amplitudes per MPI rank.");
                }


                std::string pluginGetName() const override
                {
                    return pluginName;
                }


                void setMappingDescription(MappingDesc* cellDescription) override
                {
                    this->cellDescription = cellDescription;
                }


                void restart(uint32_t timeStep, const std::string restartDirectory) override
                {
                    // only load backup if radiation is calculated:
                    if(notifyPeriod.empty())
                        return;

                    if(isMaster)
                    {
                        // this will lead to wrong lastRad output right after the checkpoint if the restart point is
                        // not a dump point. The correct lastRad data can be reconstructed from openPMD data
                        // since text based lastRad output will be obsolete soon, this is not a problem

                        readOpenPMDfile(
                            timeSumArray,
                            restartDirectory + "/" + speciesName + std::string("_radRestart_"),
                            timeStep,
                            *openPMDExtensionCheckpointing.begin() == '.' ? openPMDExtensionCheckpointing
                                                                          : '.' + openPMDExtensionCheckpointing);
                        log<radLog::SIMULATION_STATE>("Radiation (%1%): restart finished") % speciesName;
                    }
                }


                void checkpoint(uint32_t timeStep, const std::string restartDirectory)
                {
                    // only write backup if radiation is calculated:
                    if(notifyPeriod.empty())
                        return;

                    // collect data GPU -> CPU -> Master
                    copyRadiationDeviceToHost();
                    collectRadiationOnMaster();
                    sumAmplitudesOverTime(tmp_result, timeSumArray);

                    /*
                     * No need to keep the Series open for checkpointing, so
                     * just quickly open it here.
                     */
                    std::optional<::openPMD::Series> tmp;
                    writeOpenPMDfile(
                        tmp_result,
                        restartDirectory,
                        speciesName + std::string("_radRestart"),
                        WriteOpenPMDParams{
                            tmp,
                            "_%T_0_0_0"
                                + (*openPMDExtensionCheckpointing.begin() == '.'
                                       ? openPMDExtensionCheckpointing
                                       : '.' + openPMDExtensionCheckpointing),
                            openPMDCheckpointConfig});
                }


            private:
                /**
                 * The plugin is loaded on every MPI rank, and therefor this function is
                 * executed on every MPI rank.
                 * One host with MPI rank 0 is defined to be the master.
                 * It creates a folder where all the
                 * results are saved and, depending on the type of radiation calculation,
                 * creates an additional data structure for the summation of all
                 * intermediate values.
                 * On every host data structure for storage of the calculated radiation
                 * is created.       */
                void pluginLoad() override
                {
                    if(!notifyPeriod.empty())
                    {
                        if(numJobs <= 0)
                        {
                            std::cerr << "'numJobs' must be '>=1' value is adjusted from" << numJobs << " to '1'."
                                      << std::endl;
                            numJobs = 1;
                        }
                        /* allocate memory for all amplitudes for temporal data collection
                         * ACCUMULATOR! Should be in double precision for numerical stability.
                         */
                        tmp_result.resize(elements_amplitude(), Amplitude::zero());

                        /*only rank 0 creates a file*/
                        isMaster = reduce.hasResult(mpi::reduceMethods::Reduce());

                        /* Buffer for GPU results.
                         * The second dimension is used to store intermediate results if command
                         * line option numJobs is > 1.
                         */
                        radiation
                            = std::make_unique<GridBuffer<Amplitude, 2>>(DataSpace<2>(elements_amplitude(), numJobs));

                        freqInit.Init(frequencies_from_list::listLocation);
                        freqFkt = freqInit.getFunctor();

                        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
                        pmacc::Filesystem<simDim>& fs = Environment<simDim>::get().Filesystem();

                        if(isMaster)
                        {
                            timeSumArray.resize(elements_amplitude(), Amplitude::zero());

                            /* save detector position / observation direction */
                            detectorPositions.resize(parameters::N_observer);
                            for(uint32_t detectorIndex = 0; detectorIndex < parameters::N_observer; ++detectorIndex)
                            {
                                detectorPositions[detectorIndex]
                                    = radiation_observer::observation_direction(detectorIndex);
                            }

                            /* save detector frequencies */
                            detectorFrequencies.resize(radiation_frequencies::N_omega);
                            for(uint32_t detectorIndex = 0; detectorIndex < radiation_frequencies::N_omega;
                                ++detectorIndex)
                            {
                                detectorFrequencies[detectorIndex] = freqFkt.get(detectorIndex);
                            }
                        }

                        if(isMaster)
                        {
                            fs.createDirectory("radiationOpenPMD");
                            fs.setDirectoryPermissions("radiationOpenPMD");
                        }


                        if(isMaster && radPerGPU)
                        {
                            fs.createDirectory(folderRadPerGPU);
                            fs.setDirectoryPermissions(folderRadPerGPU);
                        }

                        if(isMaster && totalRad)
                        {
                            // create folder for total output
                            fs.createDirectory(folderTotalRad);
                            fs.setDirectoryPermissions(folderTotalRad);
                        }
                        if(isMaster && lastRad)
                        {
                            // create folder for total output
                            fs.createDirectory(folderLastRad);
                            fs.setDirectoryPermissions(folderLastRad);
                        }
                    }
                }


                void pluginUnload() override
                {
                    if(!notifyPeriod.empty())
                    {
                        // Some funny things that make it possible for the kernel to calculate
                        // the absolute position of the particles
                        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                        DataSpace<simDim> localSize(subGrid.getLocalDomain().size);
                        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                        DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
                        globalOffset.y() += (localSize.y() * numSlides);

                        // only print data at end of simulation if no dump period was set
                        if(dumpPeriod == 0)
                        {
                            collectDataGPUToMaster();
                            writeAllFiles(globalOffset);
                        }

                        CUDA_CHECK(cuplaGetLastError());
                    }
                }


                /** Method to copy data from GPU to CPU */
                void copyRadiationDeviceToHost()
                {
                    radiation->deviceToHost();
                    eventSystem::getTransactionEvent().waitForFinished();

                    auto dbox = radiation->getHostBuffer().getDataBox();
                    int numAmp = elements_amplitude();
                    // update the main result matrix (y index zero)
                    for(int resultIdx = 1; resultIdx < numJobs; ++resultIdx)
                        for(int ampIdx = 0; ampIdx < numAmp; ++ampIdx)
                        {
                            dbox(DataSpace<2>(ampIdx, 0)) += dbox(DataSpace<2>(ampIdx, resultIdx));
                        }
                }


                /** write radiation from each GPU to file individually
                 *  requires call of copyRadiationDeviceToHost() before */
                void saveRadPerGPU(const DataSpace<simDim> currentGPUpos)
                {
                    if(radPerGPU)
                    {
                        // only print lastGPUrad if full time period was covered
                        if(lastGPUpos == currentGPUpos)
                        {
                            std::stringstream last_time_step_str;
                            std::stringstream current_time_step_str;
                            std::stringstream GPUpos_str;

                            last_time_step_str << lastStep;
                            current_time_step_str << currentStep;

                            for(uint32_t dimIndex = 0; dimIndex < simDim; ++dimIndex)
                                GPUpos_str << "_" << currentGPUpos[dimIndex];

                            writeFile(
                                radiation->getHostBuffer().getBasePointer(),
                                folderRadPerGPU + "/" + speciesName + "_radPerGPU_pos" + GPUpos_str.str() + "_time_"
                                    + last_time_step_str.str() + "-" + current_time_step_str.str() + ".dat");
                        }
                        lastGPUpos = currentGPUpos;
                    }
                }


                /** returns number of observers (radiation detectors) */
                static unsigned int elements_amplitude()
                {
                    return radiation_frequencies::N_omega
                        * parameters::N_observer; // storage for amplitude results on GPU
                }


                /** combine radiation data from each CPU and store result on master
                 *  copyRadiationDeviceToHost() should be called before */
                void collectRadiationOnMaster()
                {
                    reduce(
                        pmacc::math::operation::Add(),
                        tmp_result.data(),
                        radiation->getHostBuffer().getBasePointer(),
                        elements_amplitude(),
                        mpi::reduceMethods::Reduce());
                }


                /** add collected radiation data to previously stored data
                 *  should be called after collectRadiationOnMaster() */
                void sumAmplitudesOverTime(std::vector<Amplitude>& targetArray, std::vector<Amplitude>& summandArray)
                {
                    if(isMaster)
                    {
                        // add last amplitudes to previous amplitudes
                        for(unsigned int i = 0; i < elements_amplitude(); ++i)
                            targetArray[i] += summandArray[i];
                    }
                }


                /** writes to file the emitted radiation only from the current
                 *  time step. Radiation from previous time steps is neglected. */
                void writeLastRadToText()
                {
                    // only the master rank writes data
                    if(isMaster)
                    {
                        // write file only if lastRad flag was selected
                        if(lastRad)
                        {
                            // get time step as string
                            std::stringstream o_step;
                            o_step << currentStep;

                            // write lastRad data to txt
                            writeFile(
                                tmp_result.data(),
                                folderLastRad + "/" + filename_prefix + "_" + o_step.str() + ".dat");
                        }
                    }
                }


                /** writes the total radiation (over entire simulation time) to file */
                void writeTotalRadToText()
                {
                    // only the master rank writes data
                    if(isMaster)
                    {
                        // write file only if totalRad flag was selected
                        if(totalRad)
                        {
                            // get time step as string
                            std::stringstream o_step;
                            o_step << currentStep;

                            // write totalRad data to txt
                            writeFile(
                                timeSumArray.data(),
                                folderTotalRad + "/" + filename_prefix + "_" + o_step.str() + ".dat");
                        }
                    }
                }


                /** write total radiation data as openPMD file */
                void writeAmplitudesToOpenPMD()
                {
                    writeOpenPMDfile(
                        timeSumArray,
                        std::string("radiationOpenPMD/"),
                        speciesName + std::string("_radAmplitudes"),
                        WriteOpenPMDParams{m_series, openPMDSuffix, openPMDConfig});
                }


                /** perform all operations to get data from GPU to master */
                void collectDataGPUToMaster()
                {
                    // collect data GPU -> CPU -> Master
                    copyRadiationDeviceToHost();
                    collectRadiationOnMaster();
                    sumAmplitudesOverTime(timeSumArray, tmp_result);
                }


                /** write all possible/selected output */
                void writeAllFiles(const DataSpace<simDim> currentGPUpos)
                {
                    // write data to files
                    saveRadPerGPU(currentGPUpos);
                    writeLastRadToText();
                    writeTotalRadToText();
                    writeAmplitudesToOpenPMD();
                }


                /** This method returns openPMD data structure names for amplitudes
                 *
                 *  Arguments:
                 *  int index - index of Amplitude
                 *              "-1" return record name
                 *
                 *  Return:
                 *  std::string - name
                 *
                 * This method avoids initializing static constexpr string arrays.
                 */
                static const std::string dataLabels(int index)
                {
                    const std::string path("Amplitude");

                    /* return record name if handed -1 */
                    if(index == -1)
                        return path;

                    const std::string dataLabelsList[] = {"x_Re", "x_Im", "y_Re", "y_Im", "z_Re", "z_Im"};

                    return dataLabelsList[index];
                }

                /** This method returns openPMD data structure names for detector directions
                 *
                 *  Arguments:
                 *  int index - index of detector
                 *              "-1" return record name
                 *
                 *  Return:
                 *  std::string - name
                 *
                 * This method avoids initializing static const string arrays.
                 */
                static const std::string dataLabelsDetectorDirection(int index)
                {
                    const std::string path("DetectorDirection");

                    /* return record name if handed -1 */
                    if(index == -1)
                        return path;

                    const std::string dataLabelsList[] = {"x", "y", "z"};

                    return dataLabelsList[index];
                }


                /** This method returns openPMD data structure names for detector frequencies
                 *
                 *  Arguments:
                 *  int index - index of detector
                 *              "-1" return record name
                 *
                 *  Return:
                 *  std::string - name
                 *
                 * This method avoids initializing static const string arrays.
                 */
                static const std::string dataLabelsDetectorFrequency(int index)
                {
                    const std::string path("DetectorFrequency");

                    /* return record name if handed -1 */
                    if(index == -1)
                        return path;

                    const std::string dataLabelsList[] = {"omega"};

                    return dataLabelsList[index];
                }

                /** This method returns openPMD data structure names for all mesh records
                 *
                 *  Arguments:
                 *  int index - index of record
                 *              "-1" return number of mesh records
                 *
                 *  Return:
                 *  std::string - name
                 *
                 * This method avoids initializing static const string arrays.
                 */
                static const std::string meshRecordLabels(int index)
                {
                    if(index == idLabels::Amplitude)
                        return dataLabels(-1);
                    else if(index == idLabels::Detector)
                        return dataLabelsDetectorDirection(-1);
                    else if(index == idLabels::Frequency)
                        return dataLabelsDetectorFrequency(-1);
                    else
                        return std::string("this-record-does-not-exist");
                }

                /*
                 * These are the params that are different between checkpointing and regular output.
                 */
                struct WriteOpenPMDParams
                {
                    // Series to be used, maybe already initialized
                    std::optional<::openPMD::Series>& series;
                    std::string const& extension;
                    std::string const& jsonConfig;
                };

                /** Write Amplitude data to openPMD file
                 *
                 * Arguments:
                 * Amplitude* values - array of complex amplitude values
                 * std::string const & path - directory for data storage
                 * std::string const & name - file name to store data to
                 * WriteOpenPMDParams params - See struct description
                 */
                void writeOpenPMDfile(
                    std::vector<Amplitude>& values,
                    std::string const& dir,
                    std::string const& name,
                    WriteOpenPMDParams const& params)
                {
                    auto const& [series, extension, jsonConfig] = params;
                    std::ostringstream filename;
                    if(std::any_of(extension.begin(), extension.end(), [](char const c) { return c == '.'; }))
                    {
                        filename << name << extension;
                    }
                    else
                    {
                        filename << name << '.' << extension;
                    }

                    if(!series.has_value())
                    {
                        GridController<simDim>& gc = Environment<simDim>::get().GridController();
                        auto communicator = gc.getCommunicator().getMPIComm();
                        series = std::make_optional(::openPMD::Series(
                            dir + '/' + filename.str(),
                            ::openPMD::Access::CREATE,
                            communicator,
                            jsonConfig));

                        /* begin recommended openPMD global attributes */
                        series->setMeshesPath(meshesPathName);
                        const std::string software("PIConGPU");
                        std::stringstream softwareVersion;
                        softwareVersion << PICONGPU_VERSION_MAJOR << "." << PICONGPU_VERSION_MINOR << "."
                                        << PICONGPU_VERSION_PATCH;
                        if(!std::string(PICONGPU_VERSION_LABEL).empty())
                            softwareVersion << "-" << PICONGPU_VERSION_LABEL;
                        series->setSoftware(software, softwareVersion.str());

                        std::string author = Environment<>::get().SimulationDescription().getAuthor();
                        if(author.length() > 0)
                            series->setAuthor(author);

                        std::string date = helper::getDateString("%F %T %z");
                        series->setDate(date);
                        /* end recommended openPMD global attributes */
                    }
                    else
                    {
                        /*
                         * Check that the filename is the same.
                         * Series::name() returns the specified path without filename extension and without the
                         * dirname, so we need to strip that information.
                         */
                        auto filename_str = filename.str();
                        auto pos = filename_str.find_last_of('.');
                        if(pos != std::string::npos)
                        {
                            filename_str = filename_str.substr(0, pos);
                        }
                        if(series->name() != filename_str)
                        {
                            /*
                             * Should normally not happen, this is just to aid debugging if it does happen anyway.
                             */
                            throw std::runtime_error(
                                "[Radiation plugin] Internal error: openPMD Series from previous run of the plugin "
                                "was initiated with file name '"
                                + series->name() + "', but now filename '" + filename_str
                                + "' is requested. Aborting.");
                        }
                    }
                    ::openPMD::Series& openPMDdataFile = series.value();
                    ::openPMD::Iteration openPMDdataFileIteration = openPMDdataFile.writeIterations()[currentStep];

                    /* begin required openPMD global attributes */
                    openPMDdataFileIteration.setDt<float_X>(DELTA_T);
                    const float_X time = float_X(currentStep) * DELTA_T;
                    openPMDdataFileIteration.setTime(time);
                    openPMDdataFileIteration.setTimeUnitSI(UNIT_TIME);
                    /* end required openPMD global attributes */

                    // begin: write per-rank amplitude data
                    if(writeDistributedAmplitudes)
                    {
                        ::openPMD::Mesh mesh_amp = openPMDdataFileIteration.meshes["Amplitude_distributed"];

                        mesh_amp.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default
                        mesh_amp.setDataOrder(::openPMD::Mesh::DataOrder::C);
                        mesh_amp.setGridSpacing(std::vector<double>{1.0, 1.0, 1.0});
                        mesh_amp.setGridGlobalOffset(std::vector<double>{0.0, 0.0, 0.0});
                        mesh_amp.setGridUnitSI(1.0);
                        mesh_amp.setAxisLabels(
                            std::vector<std::string>{"MPI", "detector direction index", "detector frequency index"});
                        mesh_amp.setUnitDimension(std::map<::openPMD::UnitDimension, double>{
                            {::openPMD::UnitDimension::L, 2.0},
                            {::openPMD::UnitDimension::M, 1.0},
                            {::openPMD::UnitDimension::T, -1.0}});

                        /* get the radiation amplitude unit */
                        Amplitude UnityAmplitude(1., 0., 0., 0., 0., 0.);
                        const picongpu::float_64 factor = UnityAmplitude.calc_radiation() * UNIT_ENERGY * UNIT_TIME;

                        // buffer for data re-arangement
                        const int N_tmpBuffer = radiation_frequencies::N_omega * parameters::N_observer;
                        std::vector<std::complex<float_64>> fallbackBuffer;

                        // reshape abstract MeshRecordComponent
                        ::openPMD::Datatype datatype_amp = ::openPMD::determineDatatype<std::complex<float_64>>();
                        auto communicator = Environment<simDim>::get().GridController().getCommunicator();
                        ::openPMD::Extent total_extent_amp
                            = {size_t(communicator.getSize()), parameters::N_observer, radiation_frequencies::N_omega};
                        ::openPMD::Offset local_offset_amp = {size_t(communicator.getRank()), 0, 0};
                        ::openPMD::Extent local_extent_amp
                            = {1, parameters::N_observer, radiation_frequencies::N_omega};

                        auto srcBuffer = radiation->getHostBuffer().getBasePointer();

                        /*
                         * numComponents includes the components of a complex number, e.g. in a 3D simulation,
                         * numComponents is 6.
                         * Since the distributed output uses native complex types, we don't want this.
                         * ---> Amplitude::numComponents / 2
                         */
                        for(uint32_t ampIndex = 0; ampIndex < Amplitude::numComponents / 2; ++ampIndex)
                        {
                            constexpr char const* labels[] = {"x", "y", "z"};
                            std::string dir = labels[ampIndex];
                            mesh_amp[dir].setUnitSI(factor);
                            mesh_amp[dir].setPosition(std::vector<double>{0.0, 0.0, 0.0});
                            ::openPMD::Dataset dataset_amp = ::openPMD::Dataset(datatype_amp, total_extent_amp);
                            mesh_amp[dir].resetDataset(dataset_amp);

                            // ask openPMD to create a buffer for us
                            // in some backends (ADIOS2), this allows avoiding memcopies
                            auto span = ::picongpu::openPMD::storeChunkSpan<std::complex<double>>(
                                            mesh_amp[dir],
                                            local_offset_amp,
                                            local_extent_amp,
                                            [&fallbackBuffer](size_t numElements)
                                            {
                                                // if there is no special backend support for creating buffers,
                                                // use the fallback buffer
                                                fallbackBuffer.resize(numElements);
                                                return std::shared_ptr<std::complex<float_64>>{
                                                    fallbackBuffer.data(),
                                                    [](auto const*) {}};
                                            })
                                            .currentBuffer();

                            // std::complex has guarantees on array-oriented access, so let's use this to make our
                            // lives easer
                            auto const* srcBuffer_reinterpreted
                                = reinterpret_cast<std::complex<picongpu::float_64>*>(srcBuffer);
                            for(uint32_t copyIndex = 0; copyIndex < N_tmpBuffer; ++copyIndex)
                            {
                                span[copyIndex]
                                    = srcBuffer_reinterpreted[ampIndex + (Amplitude::numComponents / 2) * copyIndex];
                            }
                            // flush data now
                            // this allows us to reuse the fallbackBuffer in the next iteration
                            openPMDdataFile.flush();
                        }
                    }
                    // end: write per-rank amplitude data

                    // begin: write amplitude data
                    {
                        ::openPMD::Mesh mesh_amp = openPMDdataFileIteration.meshes[dataLabels(-1)];

                        mesh_amp.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default
                        mesh_amp.setDataOrder(::openPMD::Mesh::DataOrder::C);
                        mesh_amp.setGridSpacing(std::vector<double>{1.0, 1.0, 1.0});
                        mesh_amp.setGridGlobalOffset(std::vector<double>{0.0, 0.0, 0.0});
                        mesh_amp.setGridUnitSI(1.0);
                        mesh_amp.setAxisLabels(
                            std::vector<std::string>{"detector direction index", "detector frequency index", "None"});
                        mesh_amp.setUnitDimension(std::map<::openPMD::UnitDimension, double>{
                            {::openPMD::UnitDimension::L, 2.0},
                            {::openPMD::UnitDimension::M, 1.0},
                            {::openPMD::UnitDimension::T, -1.0}});

                        /* get the radiation amplitude unit */
                        Amplitude UnityAmplitude(1., 0., 0., 0., 0., 0.);
                        const picongpu::float_64 factor = UnityAmplitude.calc_radiation() * UNIT_ENERGY * UNIT_TIME;

                        // buffer for data re-arangement
                        const int N_tmpBuffer = radiation_frequencies::N_omega * parameters::N_observer;
                        std::vector<float_64> fallbackBuffer;

                        // reshape abstract MeshRecordComponent
                        ::openPMD::Datatype datatype_amp = ::openPMD::determineDatatype<float_64>();
                        ::openPMD::Extent extent_amp = {parameters::N_observer, radiation_frequencies::N_omega, 1};
                        ::openPMD::Offset offset_amp = {0, 0, 0};

                        for(uint32_t ampIndex = 0; ampIndex < Amplitude::numComponents; ++ampIndex)
                        {
                            std::string dir = dataLabels(ampIndex);
                            mesh_amp[dir].setUnitSI(factor);
                            mesh_amp[dir].setPosition(std::vector<double>{0.0, 0.0, 0.0});
                            ::openPMD::Dataset dataset_amp = ::openPMD::Dataset(datatype_amp, extent_amp);
                            mesh_amp[dir].resetDataset(dataset_amp);

                            // ask openPMD to create a buffer for us
                            // in some backends (ADIOS2), this allows avoiding memcopies
                            if(isMaster)
                            {
                                auto span
                                    = ::picongpu::openPMD::storeChunkSpan<double>(
                                          mesh_amp[dir],
                                          offset_amp,
                                          extent_amp,
                                          [&fallbackBuffer](size_t numElements)
                                          {
                                              // if there is no special backend support for creating buffers,
                                              // use the fallback buffer
                                              fallbackBuffer.resize(numElements);
                                              return std::shared_ptr<float_64>{fallbackBuffer.data(), [](auto const*) {
                                                                               }};
                                          })
                                          .currentBuffer();

                                // select data
                                for(uint32_t copyIndex = 0; copyIndex < N_tmpBuffer; ++copyIndex)
                                {
                                    span[copyIndex] = reinterpret_cast<picongpu::float_64*>(
                                        values.data())[ampIndex + Amplitude::numComponents * copyIndex];
                                }
                            }
                            // flush data now
                            // this allows us to reuse the fallbackBuffer in the next iteration
                            openPMDdataFile.flush();
                        }
                    }
                    // end: write amplitude data


                    // start: write observer
                    // prepare n meshes
                    ::openPMD::Mesh mesh_n = openPMDdataFileIteration.meshes[dataLabelsDetectorDirection(-1)];
                    // write mesh attributes
                    mesh_n.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default
                    mesh_n.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    mesh_n.setGridSpacing(std::vector<double>{1.0, 1.0, 1.0});
                    mesh_n.setGridGlobalOffset(std::vector<double>{0.0, 0.0, 0.0});
                    mesh_n.setGridUnitSI(1.0);
                    mesh_n.setAxisLabels(std::vector<std::string>{"detector direction index", "None", "None"});
                    mesh_n.setUnitDimension(std::map<::openPMD::UnitDimension, double>{});

                    {
                        std::vector<float_64> fallbackBuffer;

                        // reshape abstract MeshRecordComponent
                        ::openPMD::Datatype datatype_n = ::openPMD::determineDatatype<float_64>();
                        ::openPMD::Extent extent_n = {parameters::N_observer, 1, 1};
                        ::openPMD::Offset offset_n = {0, 0, 0};

                        const picongpu::float_64 factorDirection = 1.0;

                        for(uint32_t detectorDim = 0; detectorDim < 3; ++detectorDim)
                        {
                            std::string dir = dataLabelsDetectorDirection(detectorDim);
                            mesh_n[dir].setUnitSI(factorDirection);
                            mesh_n[dir].setPosition(std::vector<double>{0.0, 0.0, 0.0});
                            ::openPMD::Dataset dataset_n = ::openPMD::Dataset(datatype_n, extent_n);
                            mesh_n[dir].resetDataset(dataset_n);

                            if(isMaster)
                            {
                                // ask openPMD to create a buffer for us
                                // in some backends (ADIOS2), this allows avoiding memcopies
                                auto span
                                    = ::picongpu::openPMD::storeChunkSpan<double>(
                                          mesh_n[dir],
                                          offset_n,
                                          extent_n,
                                          [&fallbackBuffer](size_t numElements)
                                          {
                                              // if there is no special backend support for creating buffers,
                                              // use the fallback buffer
                                              fallbackBuffer.resize(numElements);
                                              return std::shared_ptr<float_64>{fallbackBuffer.data(), [](auto const*) {
                                                                               }};
                                          })
                                          .currentBuffer();

                                // select data
                                for(uint32_t copyIndex = 0u; copyIndex < parameters::N_observer; ++copyIndex)
                                {
                                    span[copyIndex] = reinterpret_cast<picongpu::float_64*>(
                                        detectorPositions.data())[detectorDim + 3u * copyIndex];
                                }
                            }

                            // flush data now
                            // this allows us to reuse the fallbackBuffer in the next iteration
                            openPMDdataFile.flush();
                        }
                    }

                    // end: write observer


                    // start: write frequencies
                    // prepare omega mesh
                    ::openPMD::Mesh mesh_omega = openPMDdataFileIteration.meshes[dataLabelsDetectorFrequency(-1)];
                    // write mesh attributes
                    mesh_omega.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default
                    mesh_omega.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    mesh_omega.setGridSpacing(std::vector<double>{1.0, 1.0, 1.0});
                    mesh_omega.setGridGlobalOffset(std::vector<double>{0.0, 0.0, 0.0});
                    mesh_omega.setGridUnitSI(1.0);
                    mesh_omega.setAxisLabels(std::vector<std::string>{"None", "detector frequency index", "None"});
                    std::map<::openPMD::UnitDimension, double> const unitDimensions_omega{
                        {::openPMD::UnitDimension::T, -1.0}};
                    mesh_omega.setUnitDimension(unitDimensions_omega);

                    // write mesh attributes
                    ::openPMD::MeshRecordComponent omega_mrc = mesh_omega[dataLabelsDetectorFrequency(0)];
                    const picongpu::float_64 factorOmega = 1.0 / UNIT_TIME;
                    omega_mrc.setUnitSI(factorOmega);
                    omega_mrc.setPosition(std::vector<double>{0.0, 0.0, 0.0});

                    // reshape abstract MeshRecordComponent
                    ::openPMD::Datatype datatype_omega = ::openPMD::determineDatatype<float_64>();
                    ::openPMD::Extent extent_omega = {1, radiation_frequencies::N_omega, 1};
                    ::openPMD::Dataset dataset_omega = ::openPMD::Dataset(datatype_omega, extent_omega);
                    omega_mrc.resetDataset(dataset_omega);

                    if(isMaster)
                    {
                        // write actual data
                        ::openPMD::Offset offset_omega = {0, 0, 0};
                        /*
                         * Here, we don't use storeChunkSpan, since detectorFrequencies
                         * is created and filled upon activation of the plugin,
                         * so it survives beyond the writing of a single dataset.
                         */
                        omega_mrc.storeChunk(detectorFrequencies, offset_omega, extent_omega);
                    }
                    // end: write frequencies

                    openPMDdataFileIteration.close();
                    openPMDdataFile.flush();
                }


                /** Read Amplitude data from openPMD file
                 *
                 * Arguments:
                 * Amplitude* values - array of complex amplitudes to store data in
                 * std::string name - path and beginning of file name with data stored in
                 * const int timeStep - time step to read
                 */
                void readOpenPMDfile(
                    std::vector<Amplitude>& values,
                    std::string name,
                    const int timeStep,
                    std::string const& extension)
                {
                    std::ostringstream filename;
                    /* add to standard file ending */
                    filename << name << timeStep << "_0_0_0" + extension;

                    /* check if restart file exists */
                    if(!stdfs::exists(filename.str()))
                    {
                        log<radLog::SIMULATION_STATE>(
                            "Radiation (%1%): restart file not found (%2%) - start with zero values")
                            % speciesName % filename.str();
                    }
                    else
                    {
                        ::openPMD::Series openPMDdataFile
                            = ::openPMD::Series(filename.str(), ::openPMD::Access::READ_ONLY);

                        const int N_tmpBuffer = radiation_frequencies::N_omega * parameters::N_observer;
                        picongpu::float_64* tmpBuffer = new picongpu::float_64[N_tmpBuffer];

                        for(uint32_t ampIndex = 0; ampIndex < Amplitude::numComponents; ++ampIndex)
                        {
                            ::openPMD::MeshRecordComponent dataset
                                = openPMDdataFile.iterations[timeStep]
                                      .meshes[dataLabels(-1).c_str()][dataLabels(ampIndex).c_str()];
                            ::openPMD::Extent extent = dataset.getExtent();
                            ::openPMD::Offset offset(extent.size(), 0);

                            dataset.loadChunk(std::shared_ptr<double>{tmpBuffer, [](auto const*) {}}, offset, extent);

                            openPMDdataFile.flush();

                            for(uint32_t copyIndex = 0u; copyIndex < N_tmpBuffer; ++copyIndex)
                            {
                                /* convert data directly because Amplitude is just 6 float_32 */
                                ((picongpu::float_64*) values.data())[ampIndex + Amplitude::numComponents * copyIndex]
                                    = tmpBuffer[copyIndex];
                            }
                        }

                        delete[] tmpBuffer;
                        openPMDdataFile.iterations[timeStep].close();

                        log<radLog::SIMULATION_STATE>("Radiation (%1%): read radiation data from openPMD")
                            % speciesName;
                    }
                }


                /**
                 * From the collected data from all hosts the radiated intensity is
                 * calculated by calculating the absolute value squared and multiplying
                 * this with with the appropriate physics constants.
                 * @param values
                 * @param name
                 */
                void writeFile(Amplitude* values, std::string name)
                {
                    std::ofstream outFile;
                    outFile.open(name.c_str(), std::ofstream::out | std::ostream::trunc);
                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << name << "] for output, disable plugin output. "
                                  << std::endl;
                        isMaster = false; // no Master anymore -> no process is able to write
                    }
                    else
                    {
                        for(unsigned int index_direction = 0; index_direction < parameters::N_observer;
                            ++index_direction) // over all directions
                        {
                            for(unsigned index_omega = 0; index_omega < radiation_frequencies::N_omega;
                                ++index_omega) // over all frequencies
                            {
                                // Take Amplitude for one direction and frequency,
                                // calculate the square of the absolute value
                                // and write to file.
                                outFile << values[index_omega + index_direction * radiation_frequencies::N_omega]
                                               .calc_radiation()
                                        * UNIT_ENERGY * UNIT_TIME
                                        << "\t";
                            }
                            outFile << std::endl;
                        }
                        outFile.flush();
                        outFile << std::endl; // now all data are written to file

                        if(outFile.fail())
                            std::cerr << "Error on flushing file [" << name << "]. " << std::endl;

                        outFile.close();
                    }
                }

                /**
                 * This functions calls the radiation kernel. It specifies how the
                 * calculation is parallelized.
                 *      gridDim_rad is the number of Thread-Blocks in a grid
                 *      blockDim_rad is the number of threads per block
                 *
                 * -----------------------------------------------------------
                 * | Grid                                                    |
                 * |   --------------   --------------                       |
                 * |   |   Block 0  |   |   Block 1  |                       |
                 * |   |o      o    |   |o      o    |                       |
                 * |   |o      o    |   |o      o    |                       |
                 * |   |th1    th2  |   |th1    th2  |                       |
                 * |   --------------   --------------                       |
                 * -----------------------------------------------------------
                 *
                 * !!! The TEMPLATE parameter is not used anymore.
                 * !!! But the calculations it is supposed to do is hard coded in the
                 *     kernel.
                 * !!! THIS NEEDS TO BE CHANGED !!!
                 *
                 * @param currentStep
                 */
                template<uint32_t AREA> /*This Template Parameter is not used anymore*/
                void calculateRadiationParticles(uint32_t currentStep)
                {
                    this->currentStep = currentStep;

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

                    /* execute the particle filter */
                    radiation::executeParticleFilter(particles, currentStep);

                    /* the parallelization is ONLY over directions:
                     * (a combined parallelization over direction AND frequencies
                     * turned out to be slower on GPUs of the Fermi generation (sm_2x) (couple
                     * percent) and definitely slower on Kepler GPUs (sm_3x, tested on K20))
                     */
                    const int N_observer = parameters::N_observer;
                    const auto gridDim_rad = N_observer;

                    /* number of threads per block = number of cells in a super cell
                     *          = number of particles in a Frame
                     *          (THIS IS PIConGPU SPECIFIC)
                     * A Frame is the entity that stores particles.
                     * A super cell can have many Frames.
                     * Particles in a Frame can be accessed in parallel.
                     */

                    // Some funny things that make it possible for the kernel to calculate
                    // the absolute position of the particles
                    DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
                    const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
                    globalOffset.y() += (localSize.y() * numSlides);

                    auto workerCfg = lockstep::makeWorkerCfg<ParticlesType::FrameType::frameSize>();

                    // PIC-like kernel call of the radiation kernel
                    PMACC_LOCKSTEP_KERNEL(KernelRadiationParticles{}, workerCfg)
                    (DataSpace<2>(gridDim_rad, numJobs))(
                        /*Pointer to particles memory on the device*/
                        particles->getDeviceParticlesBox(),

                        /*Pointer to memory of radiated amplitude on the device*/
                        radiation->getDeviceBuffer().getDataBox(),
                        globalOffset,
                        currentStep,
                        *cellDescription,
                        freqFkt,
                        subGrid.getGlobalDomain().size);

                    if(dumpPeriod != 0 && currentStep % dumpPeriod == 0)
                    {
                        collectDataGPUToMaster();
                        writeAllFiles(globalOffset);

                        // update time steps
                        lastStep = currentStep;

                        // reset amplitudes on GPU back to zero
                        radiation->getDeviceBuffer().reset(false);
                    }
                }
            };

        } // namespace radiation
    } // namespace plugins

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<T_Species, plugins::radiation::Radiation<T_UnspecifiedSpecies>>
            {
                using FrameType = typename T_Species::FrameType;

                // this plugin needs at least the position, a weighting, momentum and momentumPrev1 to run
                using RequiredIdentifiers = MakeSeq_t<position<>, weighting, momentum, momentumPrev1>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                using SpeciesHasMass = typename pmacc::traits::HasFlag<FrameType, massRatio<>>::type;

                using SpeciesHasCharge = typename pmacc::traits::HasFlag<FrameType, chargeRatio<>>::type;

                using type = pmacc::mp_and<SpeciesHasIdentifiers, SpeciesHasMass, SpeciesHasCharge>;
            };

        } // namespace traits
    } // namespace particles
} // namespace picongpu
