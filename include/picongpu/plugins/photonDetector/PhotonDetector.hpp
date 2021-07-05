/* Copyright 2021 Pawel Ordyna
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

#include "picongpu/param/photonDetector.param"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/common/GetOpenPMDStoredType.hpp"
#include "picongpu/plugins/multi/multi.hpp"
#include "picongpu/plugins/photonDetector/DetectParticles.kernel"
#include "picongpu/plugins/photonDetector/DetectorParams.def"
#include "picongpu/plugins/photonDetector/PhotonDetectorImpl.hpp"
#include "picongpu/plugins/photonDetector/PhotonDetectorWriter.hpp"
#include "picongpu/plugins/photonDetector/accumulation/accumulationPolicies.def"

#include <pmacc/math/operation.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/pluginSystem/INotify.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>

#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace picongpu
{
    namespace plugins
    {
        namespace photonDetector
        {
            /** PhotonDetector plugin
             *
             * A planar detector for outgoing particles
             *
             * @tparam T_DetectorConfig compile time parameters, see photonDetector.param
             * @tparam T_Species particle species to detect
             */
            template<typename T_DetectorConfig, typename T_Species>
            class PhotonDetector : public plugins::multi::ISlave
            {
            public:
                //! Slave instance description and command line options
                struct Help : public plugins::multi::IHelp
                {
                    /** creates a PhotonDetector slave plugin instance
                     *
                     * @param help plugin defined help
                     * @param id index of the plugin, range: [0;help->getNumPlugins())
                     * @param cellDescription
                     *
                     * @returns shared pointer to the slave instance with the ISlave interface
                     */
                    std::shared_ptr<ISlave> create(
                        std::shared_ptr<IHelp>& help,
                        size_t const id,
                        MappingDesc* cellDescription) override
                    {
                        return std::shared_ptr<ISlave>(
                            new PhotonDetector<T_DetectorConfig, T_Species>(help, id, cellDescription));
                    }

                    // Plugin command line options:
                    plugins::multi::Option<std::string> notifyPeriod = {"period", "notify period"};
                    plugins::multi::Option<int32_t> detectorSizeX
                        = {"size.x", "detector size, in cells, in y direction (detector coordinate system)."};
                    plugins::multi::Option<int32_t> detectorSizeY
                        = {"size.y", "detector size, in cells, in y direction (detector coordinate system)."};
                    plugins::multi::Option<std::string> placement
                        = {"placement",
                           "Placement of the detector. Simulation box side behind which the detecor is placed. "
                           "Arbitrary angles are not supported so a detector is always placed on one of the "
                           "simulation axes. Valid values are: `x-`, `x+`, `y-`, `y+`, `z-`, `z+`.  The letter is the "
                           "axis on which the detector is positioned. `-` means in front of the simulation box `+` "
                           "means behind the simulation box. So for example `x+`(`x-`) would  mean that particles"
                           "going into the positive (negative) x direction are moving towards the detector. "};
                    plugins::multi::Option<std::string> outputDir
                        = {"dir", "Directory inside simOutput where the output files are stored.", "photonDetector"};
                    plugins::multi::Option<std::string> fileName
                        = {"fileName",
                           "Output file name base. Complete file name will be: <fileName><infix>.<ext> ",
                           T_Species::FrameType::getName() + "_" + "photonDetectorData"};
                    plugins::multi::Option<std::string> fileInfix
                        = {"infix",
                           "openPMD filename infix (use to pick file- or group-based "
                           "layout in openPMD)\nSet to NULL to keep empty (e.g. to pick"
                           " group-based iteration layout). Parameter will be ignored"
                           " if a streaming backend is detected in 'ext' parameter and"
                           " an empty string will be assumed instead. "
                           "Complete file name will be: <fileName><infix>.<ext>",
                           "_%06T"};
                    plugins::multi::Option<std::string> fileExtension
                        = {"ext",
                           "openPMD filename extension (this controls the"
                           "backend picked by the openPMD API)",
                           "bp"};
                    plugins::multi::Option<std::string> jsonConfig
                        = {"json", "advanced (backend) configuration for openPMD in JSON format", "{}"};

                    ///! method used by plugin controller to get --help description
                    void registerHelp(
                        boost::program_options::options_description& desc,
                        std::string const& masterPrefix = std::string{}) override
                    {
                        notifyPeriod.registerHelp(desc, masterPrefix + prefix);
                        detectorSizeX.registerHelp(desc, masterPrefix + prefix);
                        detectorSizeY.registerHelp(desc, masterPrefix + prefix);
                        placement.registerHelp(desc, masterPrefix + prefix);
                        fileExtension.registerHelp(desc, masterPrefix + prefix);
                        jsonConfig.registerHelp(desc, masterPrefix + prefix);
                        fileInfix.registerHelp(desc, masterPrefix + prefix);
                        fileName.registerHelp(desc, masterPrefix + prefix);
                        outputDir.registerHelp(desc, masterPrefix + prefix);
                    }

                    void expandHelp(
                        boost::program_options::options_description& desc,
                        std::string const& masterPrefix = std::string{}) override
                    {
                    }

                    void validateOptions() override
                    {
                        if(notifyPeriod.size() != detectorSizeX.size())
                            throw std::runtime_error(
                                name + ": parameter detectorSizeX and period are not used the same number of times");
                        if(notifyPeriod.size() != detectorSizeY.size())
                            throw std::runtime_error(
                                name + ": parameter detectorSizeY and period are not used the same number of times");
                        if(notifyPeriod.size() != placement.size())
                            throw std::runtime_error(
                                name + ": parameter placement and period are not used the same number of times");

                        // check if user passed placements are valid
                        for(auto const& placementStr : placement)
                        {
                            std::vector<std::string> supportedPlacements{"x-", "x+", "y-", "y+", "z-", "z+"};
                            if(std::find(supportedPlacements.begin(), supportedPlacements.end(), placementStr)
                               == supportedPlacements.end())
                            {
                                throw std::runtime_error(name + ": invalid detector placement '" + placementStr + "'");
                            }
                        }
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

                    std::string const name = "photonDetector";
                    //! short description of the plugin
                    std::string const description = "Propagate particles leaving the simulation to a planar detector.";
                    //! prefix used for command line arguments
                    std::string const prefix = T_Species::FrameType::getName() + "_" + std::string("photonDetector");
                };

            private:
                using AccumulationPolicy =
                    typename T_DetectorConfig::AccumulationPolicy::template apply<T_Species>::type;
                using StoredType = typename AccumulationPolicy::Type;
                using OpenPMDStoredType = typename openPMD::GetOpenPMDStoredType<StoredType>::type;

                //! plugin help description storing command line parameters
                std::shared_ptr<Help> help_m;
                //! id of the slave plugin instance
                size_t id_m;
                //! pointer to a detector instance (detects particles, stores output, provides extra description)
                std::unique_ptr<PhotonDetectorImpl<T_DetectorConfig, T_Species>> detector;
                //! pointer to a writer for writing output
                std::unique_ptr<PhotonDetectorWriter<OpenPMDStoredType>> writer;
                // TODO: Is it ok to use the internal implementation or should I use std::vector instead?
                using BufferMaster = pmacc::HostBufferIntern<StoredType, DIM2>;
                std::unique_ptr<BufferMaster> detectorBufferMaster;
                MappingDesc cellDescription_m;
                //! the short name of the species (prefix) processed by this instance
                std::string const speciesName;
                //! defines time steps at which the output is written
                std::string notifyPeriod;

                // For collecting detector values from multiple devices on master:
                pmacc::mpi::MPIReduce reduce;
                using ReduceMethod = pmacc::mpi::reduceMethods::Reduce;

                //! exchange direction in which the detector is placed.
                DetectorPlacement placement;
                //! the size of the detector in cells
                DataSpace<DIM2> detectorSize;
                //! Is this instance running on the master  (rank 0)?
                const bool isMaster;

            public:
                /** PhotonDetector object initializer.
                 *
                 * @param help plugin defined help
                 * @param id index of the plugin, range: [0;help_m->getNumPlugins())
                 * @param cellDescription
                 */
                HINLINE PhotonDetector(
                    std::shared_ptr<plugins::multi::IHelp>& help,
                    size_t const id,
                    MappingDesc* cellDescription)
                    : help_m(std::static_pointer_cast<Help>(help))
                    , id_m(id)
                    , cellDescription_m(*cellDescription)
                    , speciesName(T_Species::FrameType::getName())
                    , isMaster(reduce.hasResult(ReduceMethod()))
                {
                    //! Register this slave plugin instance for calling notify on each step
                    Environment<>::get().PluginConnector().setNotificationPeriod(this, help_m->notifyPeriod.get(id));

                    detectorSize = {help_m->detectorSizeX.get(id), help_m->detectorSizeY.get(id)};
                    notifyPeriod = help_m->notifyPeriod.get(id);

                    std::map<std::string, DetectorPlacement> placementMap{
                        {"x-", DetectorPlacement::XFront},
                        {"x+", DetectorPlacement::XRear},
                        {"y-", DetectorPlacement::YFront},
                        {"y+", DetectorPlacement::YRear},
                        {"z-", DetectorPlacement::ZFront},
                        {"z+", DetectorPlacement::ZRear},
                    };
                    try
                    {
                        placement = placementMap.at(help_m->placement.get(id));
                    }
                    catch(std::out_of_range const& e)
                    {
                        throw PluginException(
                            "[Plugin] [" + help_m->getOptionPrefix()
                            + "] placement must be `x-`, `x+`, `y-`, `y+`, `z-`, or `z+`");
                    }

                    using DetectorImplType = PhotonDetectorImpl<T_DetectorConfig, T_Species>;
                    detector = std::make_unique<DetectorImplType>(detectorSize, placement);

                    // Set up the writer object for IO:
                    // Make sure file based iteration layout  is not used with a streaming backend
                    std::string fileInfix = help_m->fileInfix.get(id);
                    std::string fileExtension = help_m->fileExtension.get(id);
                    if(fileInfix == "NULL" || fileExtension == "sst")
                    {
                        fileInfix = "";
                    }
                    std::string fileName = help_m->fileName.get(id);
                    std::string outputDir = help_m->outputDir.get(id);
                    std::string meshName = AccumulationPolicy::getOpenPMDMeshName();

                    writer = std::make_unique<PhotonDetectorWriter<OpenPMDStoredType>>(
                        isMaster,
                        fileName,
                        fileInfix,
                        fileExtension,
                        outputDir,
                        meshName,
                        precisionCast<uint64_t>(detectorSize),
                        float2_X{
                            T_DetectorConfig::cellHeight / UNIT_LENGTH,
                            T_DetectorConfig::cellWidth / UNIT_LENGTH},
                        detector->accumHostFunctor.getUnit(),
                        detector->accumHostFunctor.getUnitDimension(),
                        detector->accumHostFunctor.getName(),
                        help_m->placement.get(id),
                        T_DetectorConfig::distance);

                    // Create a buffer that is only on host side and only on master for intermediate output storage
                    if(isMaster)
                    {
                        detectorBufferMaster = std::make_unique<BufferMaster>(detectorSize);
                        detectorBufferMaster->setValue(AccumulationPolicy::initValue);
                    }
                }

                //! get plugin help description
                static std::shared_ptr<plugins::multi::IHelp> getHelp()
                {
                    return std::shared_ptr<plugins::multi::IHelp>(new Help{});
                }

                //! PhotonDetector object destructor
                ~PhotonDetector() override
                {
                }

                //! restart the plugin from a checkpoint
                HINLINE void restart(uint32_t restartStep, std::string const& restartDirectory) override
                {
                    // can be left empty
                }

            private:
                HINLINE void writeOutput(uint32_t currentStep)
                {
                    detector->deviceToHost();
                    __getTransactionEvent().waitForFinished();

                    reduce(
                        pmacc::math::operation::Add(),
                        isMaster ? detectorBufferMaster->getDataBox().getPointer() : nullptr,
                        detector->getHostDataBox().getPointer(),
                        detectorSize.productOfComponents(),
                        ReduceMethod());
                    if(isMaster)
                    {
                        (*writer)(
                            currentStep,
                            reinterpret_cast<OpenPMDStoredType*>(detectorBufferMaster->getDataBox().getPointer()));
                        detectorBufferMaster->setValue(AccumulationPolicy::initValue);
                    }
                    // Zero the detector.
                    detector->resetDeviceBuffer();
                }

            public:
                //! create a check point for the plugin
                HINLINE void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
                {
                    /* Force output on checkpoint.
                     * This is the easiest way to avoid loosing data on checkpoint.
                     * An alternative would be to collect data on master and store it in the checkpoint directory.
                     * This would be loaded on restart and later on added to the next output.
                     */
                    writeOutput(currentStep);
                }

                /** Actions performed on every step included in the notify period
                 *
                 * @param currentStep
                 **/
                HINLINE void notify(uint32_t currentStep) override
                {
                    writeOutput(currentStep);
                }

                /** Handle particles leaving the simulation box
                 *
                 * Check if the detector is placed behind this exchange direction. If so call the DetectParticles
                 * kernel to detect outgoing particles.
                 *
                 * @param speciesName_p name of the species leaving the simulation
                 * @param direction_p direction in which the species are leaving. See pmacc::type::ExchangeType for
                 *  definition.
                 */
                HINLINE void onParticleLeave(std::string const& speciesName_p, int32_t const direction_p)
                {
                    // no need to collect particles if the plugin is not active
                    if(this->notifyPeriod.empty())
                        return;
                    // collect only particles from the species handled by this plugin
                    if(speciesName_p != speciesName)
                        return;
                    // check exchange direction
                    int32_t direction;
                    switch(placement)
                    {
                    case DetectorPlacement::XFront:
                        direction = LEFT;
                        break;
                    case DetectorPlacement::XRear:
                        direction = RIGHT;
                        break;
                    case DetectorPlacement::YFront:
                        direction = TOP;
                        break;
                    case DetectorPlacement::YRear:
                        direction = BOTTOM;
                        break;
                    case DetectorPlacement::ZFront:
                        direction = FRONT;
                        break;
                    case DetectorPlacement::ZRear:
                        direction = BACK;
                        break;
                    default:
                        throw std::runtime_error(
                            "[plugin: " + help_m->prefix + "]" + "exchange direction" + std::to_string(direction_p)
                            + " not recognized");
                    }
                    if(direction_p != direction)
                        return;

                    DataConnector& dc = Environment<simDim>::get().DataConnector();
                    auto particles = dc.get<T_Species>(speciesName, true);

                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                    ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription_m, direction);
                    const DataSpace<simDim> localOffset = Environment<simDim>::get().SubGrid().getLocalDomain().offset;

                    // unary particle functor used to detect particles
                    const auto detectParticle
                        = detector->getDetectParticle(Environment<>::get().SimulationDescription().getCurrentStep());

                    PMACC_KERNEL(DetectParticles<numWorkers>{})
                    (mapper.getGridDim(), numWorkers)(
                        particles->getDeviceParticlesBox(),
                        localOffset,
                        detectParticle,
                        detector->getDeviceDataBox(),
                        mapper);
                }
            };

        } // namespace photonDetector
    } // namespace plugins
    namespace particles
    {
        namespace traits
        {
            //! Check if a species class has the attributes and flags required for detection
            template<typename T_DetectorConfig, typename T_Species, typename T_UnspecifiedSpecies>
            struct SpeciesEligibleForSolver<
                T_Species,
                plugins::photonDetector::PhotonDetector<T_DetectorConfig, T_UnspecifiedSpecies>>
            {
                using FrameType = typename T_Species::FrameType;
                using AccumulationPolicy = typename T_DetectorConfig::AccumulationPolicy;

                // This plugin needs at least the position and weighting for raytracing
                using RequiredIdentifiers = MakeSeq_t<position<>, momentum>;
                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;
                using EligibleForRayTracing = SpeciesHasIdentifiers;

                // Check the prerequisites for accumulating the values stored on detector
                using EligibleForAccumulation =
                    typename particles::traits::SpeciesEligibleForSolver<T_Species, AccumulationPolicy>::type;
                using type = typename bmpl::and_<EligibleForRayTracing, EligibleForAccumulation>;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
