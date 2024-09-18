/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#if(ENABLE_OPENPMD == 1)

// clang-format off
#include "picongpu/simulation_defines.hpp"
#include "picongpu/param/transitionRadiation.param"
// clang-format on

#    include "picongpu/plugins/transitionRadiation/TransitionRadiation.kernel"

#    include "picongpu/particles/filter/filter.hpp"
#    include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#    include "picongpu/plugins/ILightweightPlugin.hpp"
#    include "picongpu/plugins/ISimulationPlugin.hpp"
#    include "picongpu/plugins/PluginRegistry.hpp"
#    include "picongpu/plugins/common/openPMDAttributes.hpp"
#    include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#    include "picongpu/plugins/common/openPMDVersion.def"
#    include "picongpu/plugins/common/openPMDWriteMeta.hpp"
#    include "picongpu/plugins/common/stringHelpers.hpp"
#    include "picongpu/plugins/misc/misc.hpp"
#    include "picongpu/plugins/multi/multi.hpp"
#    include "picongpu/plugins/radiation/VectorTypes.hpp"
#    include "picongpu/plugins/transitionRadiation/executeParticleFilter.hpp"
#    include "picongpu/plugins/transitionRadiation/frequencies/LinearFrequencies.hpp"
#    include "picongpu/plugins/transitionRadiation/frequencies/ListFrequencies.hpp"
#    include "picongpu/plugins/transitionRadiation/frequencies/LogFrequencies.hpp"
#    include "picongpu/unitless/transitionRadiation.unitless"

#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/lockstep/lockstep.hpp>
#    include <pmacc/math/Complex.hpp>
#    include <pmacc/math/operation.hpp>
#    include <pmacc/mpi/MPIReduce.hpp>
#    include <pmacc/mpi/reduceMethods/Reduce.hpp>
#    include <pmacc/traits/HasIdentifier.hpp>

#    include <boost/filesystem.hpp>

#    include <chrono>
#    include <cmath>
#    include <cstdlib>
#    include <ctime>
#    include <fstream>
#    include <iostream>
#    include <memory>
#    include <string>
#    include <vector>


namespace picongpu
{
    namespace plugins
    {
        namespace transitionRadiation
        {
            using namespace pmacc;

            namespace po = boost::program_options;
            using complex_X = alpaka::Complex<float_X>;

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
             * @tparam ParticlesType particle type to compute transition radiation from
             */
            template<typename ParticlesType>
            class TransitionRadiation : public plugins::multi::IInstance
            {
            private:
                struct Help : public plugins::multi::IHelp
                {
                    /** creates an instance
                     *
                     * @param help plugin defined help
                     * @param id index of the plugin, range: [0;help->getNumPlugins())
                     */
                    std::shared_ptr<IInstance> create(
                        std::shared_ptr<IHelp>& help,
                        size_t const id,
                        MappingDesc* cellDescription) override
                    {
                        return std::shared_ptr<IInstance>(
                            new TransitionRadiation<ParticlesType>(help, id, cellDescription));
                    }

                    //! periodicity of computing the particle energy
                    plugins::multi::Option<std::string> notifyPeriod
                        = {"period", "enable plugin [for each n-th step]"};
                    plugins::multi::Option<std::string> optionFileName
                        = {"file", "file name to store transition radiation in: ", "transRad"};
                    plugins::multi::Option<std::string> optionFileExtention
                        = {"ext",
                           "openPMD filename extension. This controls the"
                           "backend picked by the openPMD API. Available extensions: ["
                               + openPMD::printAvailableExtensions() + "]",
                           openPMD::getDefaultExtension().c_str()};
                    plugins::multi::Option<bool> optionTextOutput
                        = {"datOutput",
                           "optional output: transition radiation as text file in readable format for the "
                           "in picongpu provided python analysis script, 1==enabled",
                           0};
                    plugins::multi::Option<float_X> foilPositionYSI
                        = {"foilPositionY",
                           "optional parameter: absolute position (in y-direction) of the virtual foil to calculate "
                           "the transition radiation for [in meter]. 0==disabled",
                           0};

                    ///! method used by plugin controller to get --help description
                    void registerHelp(
                        boost::program_options::options_description& desc,
                        std::string const& masterPrefix = std::string{}) override
                    {
                        notifyPeriod.registerHelp(desc, masterPrefix + prefix);
                        optionFileName.registerHelp(desc, masterPrefix + prefix);
                        optionFileExtention.registerHelp(desc, masterPrefix + prefix);
                        optionTextOutput.registerHelp(desc, masterPrefix + prefix);
                        foilPositionYSI.registerHelp(desc, masterPrefix + prefix);
                    }

                    void expandHelp(
                        boost::program_options::options_description& desc,
                        std::string const& masterPrefix = std::string{}) override
                    {
                    }


                    void validateOptions() override
                    {
                        ///@todo verify options
                    }

                    size_t getNumPlugins() const override
                    {
                        return notifyPeriod.size();
                    }

                    std::string getDescription() const override
                    {
                        return description;
                    }

                    std::string getOptionPrefix() const
                    {
                        return prefix;
                    }

                    std::string getName() const override
                    {
                        return name;
                    }

                    std::string const name = "TransitionRadiation";
                    //! short description of the plugin
                    std::string const description
                        = "Calculate transition radiation of given specified particle species.";
                    //! prefix used for command line arguments
                    std::string const prefix = ParticlesType::FrameType::getName() + std::string("_transRad");
                };

                using SuperCellSize = MappingDesc::SuperCellSize;

                using radLog = plugins::radiation::PIConGPUVerboseRadiation;

                std::unique_ptr<GridBuffer<float_X, DIM1>> incTransRad;
                std::unique_ptr<GridBuffer<complex_X, DIM1>> cohTransRadPara;
                std::unique_ptr<GridBuffer<complex_X, DIM1>> cohTransRadPerp;
                std::unique_ptr<GridBuffer<float_X, DIM1>> numParticles;

                transitionRadiation::frequencies::InitFreqFunctor freqInit;
                transitionRadiation::frequencies::FreqFunctor freqFkt;

                std::vector<float_X> tmpITR;
                std::vector<complex_X> tmpCTRpara;
                std::vector<complex_X> tmpCTRperp;
                std::vector<float_X> tmpNum;
                std::vector<float_X> theTransRad;
                uint32_t timeStep;

                std::string pluginName;
                std::string speciesName;
                std::string pluginPrefix;
                std::string filenamePrefix;
                std::string fileExtension;

                bool isMaster = false;
                uint32_t currentStep = 0;

                mpi::MPIReduce reduce;

                MappingDesc* m_cellDescription = nullptr;
                std::shared_ptr<Help> m_help;
                size_t m_id;

                bool textOutput;
                float_X foilPositionYSI;

            public:
                //! Constructor
                TransitionRadiation(
                    std::shared_ptr<plugins::multi::IHelp>& help,
                    size_t const id,
                    MappingDesc* cellDescription)
                    : m_cellDescription(cellDescription)
                    , m_help(std::static_pointer_cast<Help>(help))
                    , m_id(id)
                {
                    filenamePrefix = ParticlesType::FrameType::getName() + "_" + m_help->optionFileName.get(m_id);
                    fileExtension = m_help->optionFileExtention.get(m_id);

                    foilPositionYSI = m_help->foilPositionYSI.get(m_id);
                    textOutput = m_help->optionTextOutput.get(m_id);

                    init();
                }

                ~TransitionRadiation() override = default;

                /** Plugin management
                 *
                 * Implementation of base class function. Calculates the transition radiation
                 * by calling the according function of the kernel file, writes data to a
                 * file and resets the buffers if transition radiation is calculated for
                 * multiple timesteps.
                 *
                 * @param currentStep current step of simulation
                 */
                void notify(uint32_t currentStep) override
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


                //! must be implemented by the user
                static std::shared_ptr<plugins::multi::IHelp> getHelp()
                {
                    return std::shared_ptr<plugins::multi::IHelp>(new Help{});
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
                void init()
                {
                    tmpITR.resize(elementsTransitionRadiation());
                    tmpCTRpara.resize(elementsTransitionRadiation());
                    tmpCTRperp.resize(elementsTransitionRadiation());
                    tmpNum.resize(elementsTransitionRadiation());

                    /*only rank 0 create a file*/
                    isMaster = reduce.hasResult(mpi::reduceMethods::Reduce());

                    Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(m_id));

                    incTransRad
                        = std::make_unique<GridBuffer<float_X, DIM1>>(DataSpace<DIM1>(elementsTransitionRadiation()));
                    cohTransRadPara = std::make_unique<GridBuffer<complex_X, DIM1>>(
                        DataSpace<DIM1>(elementsTransitionRadiation()));
                    cohTransRadPerp = std::make_unique<GridBuffer<complex_X, DIM1>>(
                        DataSpace<DIM1>(elementsTransitionRadiation()));
                    numParticles
                        = std::make_unique<GridBuffer<float_X, DIM1>>(DataSpace<DIM1>(elementsTransitionRadiation()));

                    freqInit.Init(listFrequencies::listLocation);
                    freqFkt = freqInit.getFunctor();

                    if(isMaster)
                    {
                        theTransRad.resize(elementsTransitionRadiation());
                        for(unsigned int i = 0; i < elementsTransitionRadiation(); ++i)
                        {
                            theTransRad[i] = 0;
                        }
                    }
                }

                //! Moves transition radiation data from GPUs to CPUs.
                void copyRadiationDeviceToHost()
                {
                    incTransRad->deviceToHost();
                    eventSystem::getTransactionEvent().waitForFinished();
                    cohTransRadPara->deviceToHost();
                    eventSystem::getTransactionEvent().waitForFinished();
                    cohTransRadPerp->deviceToHost();
                    eventSystem::getTransactionEvent().waitForFinished();
                    numParticles->deviceToHost();
                    eventSystem::getTransactionEvent().waitForFinished();
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
                        pmacc::math::operation::Add(),
                        tmpITR.data(),
                        incTransRad->getHostBuffer().data(),
                        elementsTransitionRadiation(),
                        mpi::reduceMethods::Reduce());
                    reduce(
                        pmacc::math::operation::Add(),
                        tmpCTRpara.data(),
                        cohTransRadPara->getHostBuffer().data(),
                        elementsTransitionRadiation(),
                        mpi::reduceMethods::Reduce());
                    reduce(
                        pmacc::math::operation::Add(),
                        tmpCTRperp.data(),
                        cohTransRadPerp->getHostBuffer().data(),
                        elementsTransitionRadiation(),
                        mpi::reduceMethods::Reduce());
                    reduce(
                        pmacc::math::operation::Add(),
                        tmpNum.data(),
                        numParticles->getHostBuffer().data(),
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
                        if(textOutput)
                        {
                            writeFile(theTransRad.data(), filenamePrefix + "_" + o_step.str() + ".dat");
                        }
                        writeOpenPMDFile(currentStep);
                    }
                }


                //! perform all operations to get data from GPU to master
                void collectDataGPUToMaster()
                {
                    // collect data GPU -> CPU -> Master
                    copyRadiationDeviceToHost();
                    collectRadiationOnMaster();
                    sumTransitionRadiation(
                        theTransRad.data(),
                        tmpITR.data(),
                        tmpCTRpara.data(),
                        tmpCTRperp.data(),
                        tmpNum.data());
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
                            const float_X ctrPara = pmacc::math::norm(ctrParaArray[i]);
                            const float_X ctrPerp = pmacc::math::norm(ctrPerpArray[i]);
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
                                constexpr float_X transRadUnit = sim.si.getElectronCharge()
                                    * sim.si.getElectronCharge()
                                    * (1.0 / (4 * PI * sim.si.getEps0() * PI * PI * sim.si.getSpeedOfLight()));
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

                void writeOpenPMDFile(uint32_t currentStep)
                {
                    std::stringstream filename;
                    filename << filenamePrefix << "_%T." << fileExtension;

                    ::openPMD::Series series(filename.str(), ::openPMD::Access::CREATE);

                    ::openPMD::Extent extent
                        = {static_cast<unsigned long int>(transitionRadiation::frequencies::nOmega),
                           static_cast<unsigned long int>(parameters::nPhi),
                           static_cast<unsigned long int>(parameters::nTheta)};
                    ::openPMD::Offset offset = {0, 0, 0};
                    ::openPMD::Datatype datatype = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Dataset dataset{datatype, extent};

                    auto iteration = series.writeIterations()[currentStep];
                    auto mesh = iteration.meshes["transitionradiation"];

                    mesh.setAxisLabels(std::vector<std::string>{"omega index", "phi index", "theta index"});
                    mesh.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    mesh.setGridUnitSI(1);
                    mesh.setGridSpacing(std::vector<double>{1, 1, 1});
                    mesh.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default
                    mesh.setAttribute<float_X>("foilPositionY", foilPositionYSI);

                    mesh.setUnitDimension(std::map<::openPMD::UnitDimension, double>{
                        {::openPMD::UnitDimension::L, 2.0},
                        {::openPMD::UnitDimension::M, 1.0},
                        {::openPMD::UnitDimension::T, -1.0}});

                    auto transitionRadiation = mesh[::openPMD::RecordComponent::SCALAR];
                    transitionRadiation.resetDataset(dataset);

                    transitionRadiation.setUnitSI(
                        sim.si.getElectronCharge() * sim.si.getElectronCharge()
                        * (1.0 / (4 * PI * sim.si.getEps0() * PI * PI * sim.si.getSpeedOfLight())));

                    auto span = transitionRadiation.storeChunk<float_X>(offset, extent);
                    auto spanBuffer = span.currentBuffer();

                    for(unsigned int index_direction = 0; index_direction < transitionRadiation::parameters::nObserver;
                        ++index_direction)
                    {
                        // theta
                        const int i = index_direction / parameters::nPhi;
                        // phi
                        const int j = index_direction % parameters::nPhi;

                        for(unsigned int k = 0; k < transitionRadiation::frequencies::nOmega; ++k)
                        {
                            const int index = (k * parameters::nPhi + j) * parameters::nTheta + i;
                            spanBuffer[index] = static_cast<float_X>(
                                theTransRad[index_direction * transitionRadiation::frequencies::nOmega + k]);
                        }
                    }

                    // Omega axis
                    ::openPMD::Extent extentOmega
                        = {static_cast<unsigned long int>(transitionRadiation::frequencies::nOmega), 1, 1};
                    ::openPMD::Offset offsetOmega = {0, 0, 0};
                    ::openPMD::Datatype datatypeOmega = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Dataset datasetOmega{datatypeOmega, extentOmega};

                    auto meshOmega = iteration.meshes["detector omega"];

                    meshOmega.setAxisLabels(std::vector<std::string>{"omega", "", ""});
                    meshOmega.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    meshOmega.setGridUnitSI(1);
                    meshOmega.setGridSpacing(std::vector<double>{1, 1, 1});
                    meshOmega.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default

                    meshOmega.setUnitDimension(
                        std::map<::openPMD::UnitDimension, double>{{::openPMD::UnitDimension::T, -1.0}});

                    auto omega = meshOmega[::openPMD::RecordComponent::SCALAR];
                    omega.resetDataset(datasetOmega);

                    omega.setUnitSI(1.0);

                    auto spanOmega = omega.storeChunk<float_X>(offsetOmega, extentOmega);
                    auto spanBufferOmega = spanOmega.currentBuffer();

                    for(unsigned int i = 0; i < transitionRadiation::frequencies::nOmega; ++i)
                    {
                        spanBufferOmega[i] = static_cast<float_X>(freqFkt(i));
                    }

                    // Phi axis
                    ::openPMD::Extent extentPhi
                        = {1, static_cast<unsigned long int>(transitionRadiation::parameters::nPhi), 1};
                    ::openPMD::Offset offsetPhi = {0, 0, 0};
                    ::openPMD::Datatype datatypePhi = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Dataset datasetPhi{datatypePhi, extentPhi};

                    auto meshPhi = iteration.meshes["detector phi"];

                    meshPhi.setAxisLabels(std::vector<std::string>{"", "phi", ""});
                    meshPhi.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    meshPhi.setGridUnitSI(1);
                    meshPhi.setGridSpacing(std::vector<double>{1, 1, 1});
                    meshPhi.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default

                    meshPhi.setUnitDimension(std::map<::openPMD::UnitDimension, double>{});

                    auto phi = meshPhi[::openPMD::RecordComponent::SCALAR];
                    phi.resetDataset(datasetPhi);

                    phi.setUnitSI(1.0);

                    auto spanPhi = phi.storeChunk<float_X>(offsetPhi, extentPhi);
                    auto spanBufferPhi = spanPhi.currentBuffer();

                    if(transitionRadiation::parameters::nPhi > 1)
                    {
                        for(unsigned int i = 0; i < transitionRadiation::parameters::nPhi; ++i)
                        {
                            spanBufferPhi[i] = parameters::phiMin
                                + i * (parameters::phiMax - parameters::phiMin) / (parameters::nPhi - 1.0);
                        }
                    }
                    else
                    {
                        spanBufferPhi[0] = parameters::phiMin;
                    }

                    // Theta axis
                    ::openPMD::Extent extentTheta = {1, 1, transitionRadiation::parameters::nTheta};
                    ::openPMD::Offset offsetTheta = {0, 0, 0};
                    ::openPMD::Datatype datatypeTheta = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Dataset datasetTheta{datatypeTheta, extentTheta};

                    auto meshTheta = iteration.meshes["detector theta"];

                    meshTheta.setAxisLabels(std::vector<std::string>{"", "", "theta"});
                    meshTheta.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    meshTheta.setGridUnitSI(1);
                    meshTheta.setGridSpacing(std::vector<double>{1, 1, 1});
                    meshTheta.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default

                    meshTheta.setUnitDimension(std::map<::openPMD::UnitDimension, double>{});

                    auto theta = meshTheta[::openPMD::RecordComponent::SCALAR];
                    theta.resetDataset(datasetTheta);

                    theta.setUnitSI(1.0);

                    auto spanTheta = theta.storeChunk<float_X>(offsetTheta, extentTheta);
                    auto spanBufferTheta = spanTheta.currentBuffer();

                    if(transitionRadiation::parameters::nTheta > 1)
                    {
                        for(unsigned int i = 0; i < transitionRadiation::parameters::nTheta; ++i)
                        {
                            spanBufferTheta[i] = parameters::thetaMin
                                + i * (parameters::thetaMax - parameters::thetaMin) / (parameters::nTheta - 1.0);
                        }
                    }
                    else
                    {
                        spanBufferTheta[0] = parameters::thetaMin;
                    }

                    series.iterations[currentStep].close();
                }


                void restart(uint32_t restartStep, std::string const& restartDirectory) override
                {
                }

                void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
                {
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
                    auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName());

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
                    DataSpace<simDim> localSize(m_cellDescription->getGridLayout().sizeWithoutGuardND());
                    const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    DataSpace<simDim> globalOffset(subGrid.getLocalDomain().offset);
                    globalOffset.y() += (localSize.y() * numSlides);

                    // PIC-like kernel call of the radiation kernel
                    PMACC_LOCKSTEP_KERNEL(KernelTransRadParticles{})
                        .config(gridDim_rad, *particles)(
                            /*Pointer to particles memory on the device*/
                            particles->getDeviceParticlesBox(),

                            /*Pointer to memory of radiated amplitude on the device*/
                            incTransRad->getDeviceBuffer().getDataBox(),
                            cohTransRadPara->getDeviceBuffer().getDataBox(),
                            cohTransRadPerp->getDeviceBuffer().getDataBox(),
                            numParticles->getDeviceBuffer().getDataBox(),
                            globalOffset,
                            *m_cellDescription,
                            freqFkt,
                            subGrid.getGlobalDomain().size,
                            foilPositionYSI / sim.unit.length());
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

                using type = pmacc::mp_and<SpeciesHasIdentifiers, SpeciesHasMass, SpeciesHasCharge, SpeciesHasMask>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu

PIC_REGISTER_SPECIES_PLUGIN(
    picongpu::plugins::multi::Master<picongpu::plugins::transitionRadiation::TransitionRadiation<boost::mpl::_1>>);
#endif
