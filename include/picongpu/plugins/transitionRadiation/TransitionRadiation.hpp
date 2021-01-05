/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz
 *                     Finn-Ole Carstens
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

#include "picongpu/plugins/transitionRadiation/TransitionRadiation.kernel"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/transitionRadiation/ExecuteParticleFilter.hpp"
#include "picongpu/plugins/common/stringHelpers.hpp"

#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/traits/HasIdentifier.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include "picongpu/plugins/transitionRadiation/frequencies/LogFrequencies.hpp"
#include "picongpu/plugins/transitionRadiation/frequencies/LinearFrequencies.hpp"
#include "picongpu/plugins/transitionRadiation/frequencies/ListFrequencies.hpp"
#include <pmacc/math/Complex.hpp>

#include <boost/filesystem.hpp>

#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include <math.h>


namespace picongpu
{
    namespace plugins
    {
        namespace transitionRadiation
        {
            using namespace pmacc;

            namespace po = boost::program_options;
            using complex_X = pmacc::math::Complex<float_X>;

            /** Implementation of transition radiation for in situ calculation in PIConGPU
             *
             * The transition radiation implemented in this plugin is based on
             * C. B. Schroeder, E. Esarey, J. van Tilborg, and W. P. Leemans:
             * Theory of coherent transition radiation generated at a plasma-vacuum interface
             * (DOI:https://doi.org/10.1103/PhysRevE.69.016501)
             *
             * Transition radiation is created by charged particles moving through an
             * interface where one medium has a different diffraction index as the other
             * medium. Since it is mostly used to analyze electron bunches, this plugin
             * assumes that the analyzed particles have the mass and charge of electrons.
             *
             * @tparam T_ParticlesType particle type to compute transition radiation from
             */
            template<typename T_ParticlesType>
            class TransitionRadiation : public ILightweightPlugin
            {
            private:
                using SuperCellSize = MappingDesc::SuperCellSize;

                using radLog = plugins::radiation::PIConGPUVerboseRadiation;

                GridBuffer<float_X, DIM1>* incTransRad = nullptr;
                GridBuffer<complex_X, DIM1>* cohTransRadPara = nullptr;
                GridBuffer<complex_X, DIM1>* cohTransRadPerp = nullptr;
                GridBuffer<float_X, DIM1>* numParticles = nullptr;

                transitionRadiation::frequencies::InitFreqFunctor freqInit;
                transitionRadiation::frequencies::FreqFunctor freqFkt;

                float_X* tmpITR = nullptr;
                complex_X* tmpCTRpara = nullptr;
                complex_X* tmpCTRperp = nullptr;
                float_X* tmpNum = nullptr;
                float_X* theTransRad = nullptr;
                MappingDesc* cellDescription = nullptr;
                std::string notifyPeriod;
                uint32_t timeStep;

                std::string speciesName;
                std::string pluginName;
                std::string pluginPrefix;
                std::string filenamePrefix;
                std::string folderTransRad;

                float3_X* detectorPositions = nullptr;
                float_X* detectorFrequencies = nullptr;

                bool isMaster = false;
                uint32_t currentStep = 0;

                mpi::MPIReduce reduce;

            public:
                //! Constructor
                TransitionRadiation()
                    : pluginName("TransitionRadiation: calculate transition radiation of species")
                    , speciesName(T_ParticlesType::FrameType::getName())
                    , pluginPrefix(speciesName + std::string("_transRad"))
                    , folderTransRad("transRad")
                    , filenamePrefix(pluginPrefix)
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                virtual ~TransitionRadiation()
                {
                }

                /** Plugin management
                 *
                 * Implementation of base class function. Calculates the transition radiation
                 * by calling the according function of the kernel file, writes data to a
                 * file and resets the buffers if transition radiation is calculated for
                 * multiple timesteps.
                 *
                 * @param currentStep current step of simulation
                 */
                void notify(uint32_t currentStep)
                {
                    log<radLog::SIMULATION_STATE>("Transition Radition (%1%): calculate time step %2% ") % speciesName
                        % currentStep;

                    resetBuffers();
                    this->currentStep = currentStep;

                    calculateTransitionRadiation(currentStep);

                    log<radLog::SIMULATION_STATE>("Transition Radition (%1%): finished time step %2% ") % speciesName
                        % currentStep;

                    collectDataGPUToMaster();
                    writeTransRadToText();

                    log<radLog::SIMULATION_STATE>("Transition Radition (%1%): printed to table %2% ") % speciesName
                        % currentStep;
                }

                /** Implementation of base class function. Registers plugin options.
                 *
                 * @param desc boost::program_options description
                 */
                void pluginRegisterHelp(po::options_description& desc)
                {
                    desc.add_options()(
                        (pluginPrefix + ".period").c_str(),
                        po::value<std::string>(&notifyPeriod),
                        "enable plugin [for each n-th step]");
                }

                /** Implementation of base class function.
                 *
                 * @return name of plugin
                 */
                std::string pluginGetName() const
                {
                    return pluginName;
                }

                /** Implementation of base class function. Sets mapping description.
                 *
                 * @param cellDescription
                 */
                void setMappingDescription(MappingDesc* cellDescription)
                {
                    this->cellDescription = cellDescription;
                }

            private:
                //! Resets buffers for multiple transition radiation calculation per simulation.
                void resetBuffers()
                {
                    /* Resets all Databuffers and arrays for repeated calculation of the
                     * transition radiation
                     */
                    incTransRad->getDeviceBuffer().reset(false);
                    cohTransRadPara->getDeviceBuffer().reset(false);
                    cohTransRadPerp->getDeviceBuffer().reset(false);
                    numParticles->getDeviceBuffer().reset(false);

                    for(unsigned int i = 0; i < elementsTransitionRadiation(); ++i)
                    {
                        tmpITR[i] = 0;
                        tmpCTRpara[i] = 0;
                        tmpCTRperp[i] = 0;
                        tmpNum[i] = 0;
                        if(isMaster)
                        {
                            theTransRad[i] = 0;
                        }
                    }
                }

                /** Create buffers and arrays
                 *
                 * Implementation of base class function. Create buffers and arrays for
                 * transition radiation calculation and create a folder for transition
                 * radiation storage.
                 */
                void pluginLoad()
                {
                    if(!notifyPeriod.empty())
                    {
                        tmpITR = new float_X[elementsTransitionRadiation()];
                        tmpCTRpara = new complex_X[elementsTransitionRadiation()];
                        tmpCTRperp = new complex_X[elementsTransitionRadiation()];
                        tmpNum = new float_X[elementsTransitionRadiation()];

                        /*only rank 0 create a file*/
                        isMaster = reduce.hasResult(mpi::reduceMethods::Reduce());
                        pmacc::Filesystem<simDim>& fs = Environment<simDim>::get().Filesystem();

                        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

                        incTransRad = new GridBuffer<float_X, DIM1>(DataSpace<DIM1>(elementsTransitionRadiation()));
                        cohTransRadPara
                            = new GridBuffer<complex_X, DIM1>(DataSpace<DIM1>(elementsTransitionRadiation()));
                        cohTransRadPerp
                            = new GridBuffer<complex_X, DIM1>(DataSpace<DIM1>(elementsTransitionRadiation()));
                        numParticles = new GridBuffer<float_X, DIM1>(DataSpace<DIM1>(elementsTransitionRadiation()));

                        freqInit.Init(listFrequencies::listLocation);
                        freqFkt = freqInit.getFunctor();

                        if(isMaster)
                        {
                            theTransRad = new float_X[elementsTransitionRadiation()];
                            /* save detector position / observation direction */
                            detectorPositions = new float3_X[transitionRadiation::parameters::nObserver];
                            for(uint32_t detectorIndex = 0; detectorIndex < transitionRadiation::parameters::nObserver;
                                ++detectorIndex)
                            {
                                detectorPositions[detectorIndex]
                                    = transitionRadiation::observationDirection(detectorIndex);
                            }

                            /* save detector frequencies */
                            detectorFrequencies = new float_X[transitionRadiation::frequencies::nOmega];
                            for(uint32_t detectorIndex = 0; detectorIndex < transitionRadiation::frequencies::nOmega;
                                ++detectorIndex)
                            {
                                detectorFrequencies[detectorIndex] = freqFkt.get(detectorIndex);
                            }

                            for(unsigned int i = 0; i < elementsTransitionRadiation(); ++i)
                            {
                                theTransRad[i] = 0;
                            }

                            fs.createDirectory(folderTransRad);
                            fs.setDirectoryPermissions(folderTransRad);
                        }
                    }
                }

                //! Implementation of base class function. Deletes buffers andf arrays.
                void pluginUnload()
                {
                    if(!notifyPeriod.empty())
                    {
                        if(isMaster)
                        {
                            __deleteArray(theTransRad);
                        }
                        CUDA_CHECK(cuplaGetLastError());
                        __delete(incTransRad);
                        __delete(cohTransRadPara);
                        __delete(cohTransRadPerp);
                        __delete(numParticles);
                        __deleteArray(tmpITR);
                        __deleteArray(tmpCTRpara);
                        __deleteArray(tmpCTRperp);
                        __deleteArray(tmpNum);
                    }
                }

                //! Moves transition radiation data from GPUs to CPUs.
                void copyRadiationDeviceToHost()
                {
                    incTransRad->deviceToHost();
                    __getTransactionEvent().waitForFinished();
                    cohTransRadPara->deviceToHost();
                    __getTransactionEvent().waitForFinished();
                    cohTransRadPerp->deviceToHost();
                    __getTransactionEvent().waitForFinished();
                    numParticles->deviceToHost();
                    __getTransactionEvent().waitForFinished();
                }

                /** Amount of transition radiation values
                 *
                 * Calculates amount of different transition radiation values, which
                 * have to be computed.
                 *
                 * @return amount of transition radiation values to be calculated
                 */
                static unsigned int elementsTransitionRadiation()
                {
                    return transitionRadiation::frequencies::nOmega
                        * transitionRadiation::parameters::nObserver; // storage for amplitude results on GPU
                }

                /** Combine transition radiation data from each CPU and store result on master.
                 *
                 * @remark copyRadiationDeviceToHost( ) should be called before.
                 */
                void collectRadiationOnMaster()
                {
                    reduce(
                        nvidia::functors::Add(),
                        tmpITR,
                        incTransRad->getHostBuffer().getBasePointer(),
                        elementsTransitionRadiation(),
                        mpi::reduceMethods::Reduce());
                    reduce(
                        nvidia::functors::Add(),
                        tmpCTRpara,
                        cohTransRadPara->getHostBuffer().getBasePointer(),
                        elementsTransitionRadiation(),
                        mpi::reduceMethods::Reduce());
                    reduce(
                        nvidia::functors::Add(),
                        tmpCTRperp,
                        cohTransRadPerp->getHostBuffer().getBasePointer(),
                        elementsTransitionRadiation(),
                        mpi::reduceMethods::Reduce());
                    reduce(
                        nvidia::functors::Add(),
                        tmpNum,
                        numParticles->getHostBuffer().getBasePointer(),
                        elementsTransitionRadiation(),
                        mpi::reduceMethods::Reduce());
                }

                //! Write transition radiation data to file.
                void writeTransRadToText()
                {
                    // only the master rank writes data
                    if(isMaster)
                    {
                        // get time step as string
                        std::stringstream o_step;
                        o_step << currentStep;

                        // write totalRad data to txt
                        writeFile(theTransRad, folderTransRad + "/" + filenamePrefix + "_" + o_step.str() + ".dat");
                    }
                }


                //! perform all operations to get data from GPU to master
                void collectDataGPUToMaster()
                {
                    // collect data GPU -> CPU -> Master
                    copyRadiationDeviceToHost();
                    collectRadiationOnMaster();
                    sumTransitionRadiation(theTransRad, tmpITR, tmpCTRpara, tmpCTRperp, tmpNum);
                }

                /** Final transition radiation calculation on CPU side
                 *
                 * Calculate transition radiation integrals. This can't happen on the GPU
                 * since the absolute square of a sum can't be moved within a sum.
                 *
                 * @param targetArray array to store transition radiation in
                 * @param itrArray array of calculated incoherent transition radiation
                 * @param ctrParaArray array of complex values of the parallel part of the coherent transition
                 * radiation
                 * @param ctrPerpArray array of complex values of the perpendicular part of coherent transition
                 * radiation
                 * @param numArray array of amount of particles
                 */
                void sumTransitionRadiation(
                    float_X* targetArray,
                    float_X* itrArray,
                    complex_X* ctrParaArray,
                    complex_X* ctrPerpArray,
                    float_X* numArray)
                {
                    if(isMaster)
                    {
                        /************************************************************
                         ******** Here happens the true physical calculation ********
                         ************************************************************/
                        for(unsigned int i = 0; i < elementsTransitionRadiation(); ++i)
                        {
                            const float_X ctrPara = pmacc::math::abs2(ctrParaArray[i]);
                            const float_X ctrPerp = pmacc::math::abs2(ctrPerpArray[i]);
                            if(numArray[i] != 0.0)
                            {
                                targetArray[i]
                                    = (itrArray[i] + (numArray[i] - 1.0) * (ctrPara + ctrPerp) / numArray[i]);
                            }
                            else
                                targetArray[i] = 0.0;
                        }
                    }
                }

                /** Writes file with transition radiation data with the right units.
                 *
                 * @param values transition radiation values
                 * @param name name of file
                 */
                void writeFile(float_X* values, std::string name)
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
                        outFile << "# \t";
                        outFile << transitionRadiation::frequencies::getParameters();
                        outFile << transitionRadiation::parameters::nPhi << "\t";
                        outFile << transitionRadiation::parameters::phiMin << "\t";
                        outFile << transitionRadiation::parameters::phiMax << "\t";
                        outFile << transitionRadiation::parameters::nTheta << "\t";
                        outFile << transitionRadiation::parameters::thetaMin << "\t";
                        outFile << transitionRadiation::parameters::thetaMax << "\t";
                        outFile << std::endl;

                        for(unsigned int index_direction = 0;
                            index_direction < transitionRadiation::parameters::nObserver;
                            ++index_direction) // over all directions
                        {
                            for(unsigned index_omega = 0; index_omega < transitionRadiation::frequencies::nOmega;
                                ++index_omega) // over all frequencies
                            {
                                // Take Amplitude for one direction and frequency,
                                // calculate the square of the absolute value
                                // and write to file.
                                constexpr float_X transRadUnit = SI::ELECTRON_CHARGE_SI * SI::ELECTRON_CHARGE_SI
                                    * (1.0 / (4 * PI * SI::EPS0_SI * PI * PI * SI::SPEED_OF_LIGHT_SI));
                                outFile
                                    << values[index_direction * transitionRadiation::frequencies::nOmega + index_omega]
                                        * transRadUnit
                                    << "\t";

                            } // for loop over all frequencies

                            outFile << std::endl;
                        } // for loop over all frequencies

                        outFile.flush();
                        outFile << std::endl; // now all data are written to file

                        if(outFile.fail())
                            std::cerr << "Error on flushing file [" << name << "]. " << std::endl;

                        outFile.close();
                    }
                }

                /** Kernel call
                 *
                 * Executes the particle filter and calls the transition radiation kernel
                 * of the kernel file.
                 *
                 * @param currentStep current simulation iteration step
                 */
                void calculateTransitionRadiation(uint32_t currentStep)
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto particles = dc.get<T_ParticlesType>(T_ParticlesType::FrameType::getName(), true);

                    /* execute the particle filter */
                    transitionRadiation::executeParticleFilter(particles, currentStep);

                    const auto gridDim_rad = transitionRadiation::parameters::nObserver;

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

                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                    // PIC-like kernel call of the radiation kernel
                    PMACC_KERNEL(KernelTransRadParticles<numWorkers>{})
                    (gridDim_rad, numWorkers)(
                        /*Pointer to particles memory on the device*/
                        particles->getDeviceParticlesBox(),

                        /*Pointer to memory of radiated amplitude on the device*/
                        incTransRad->getDeviceBuffer().getDataBox(),
                        cohTransRadPara->getDeviceBuffer().getDataBox(),
                        cohTransRadPerp->getDeviceBuffer().getDataBox(),
                        numParticles->getDeviceBuffer().getDataBox(),
                        globalOffset,
                        *cellDescription,
                        freqFkt,
                        subGrid.getGlobalDomain().size);

                    dc.releaseData(T_ParticlesType::FrameType::getName());
                }
            };

        } // namespace transitionRadiation
    } // namespace plugins

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<
                T_Species,
                plugins::transitionRadiation::TransitionRadiation<T_UnspecifiedSpecies>>
            {
                using FrameType = typename T_Species::FrameType;

                // this plugin needs at least the weighting and momentum attributes
                using RequiredIdentifiers = MakeSeq_t<weighting, momentum, position<>>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                // this plugin needs a mass ratio for energy calculation from momentum
                using SpeciesHasMass = typename pmacc::traits::HasFlag<FrameType, massRatio<>>::type;

                // transition radiation requires charged particles
                using SpeciesHasCharge = typename pmacc::traits::HasFlag<FrameType, chargeRatio<>>::type;

                // this plugin needs the transitionRadiationMask flag
                using SpeciesHasMask = typename pmacc::traits::HasIdentifier<FrameType, transitionRadiationMask>::type;

                using type =
                    typename bmpl::and_<SpeciesHasIdentifiers, SpeciesHasMass, SpeciesHasCharge, SpeciesHasMask>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
