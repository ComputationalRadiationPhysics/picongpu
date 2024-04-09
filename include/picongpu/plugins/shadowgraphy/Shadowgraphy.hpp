/* Copyright 2023 Finn-Ole Carstens, Rene Widera
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

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/plugins/multi/multi.hpp"
#include "picongpu/plugins/shadowgraphy/ShadowgraphyHelper.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/mpi/GatherSlice.hpp>

#include <iostream>
#include <sstream>
#include <string>

#include <mpi.h>
#include <openPMD/openPMD.hpp>
#include <stdio.h>


namespace picongpu
{
    using complex_64 = alpaka::Complex<float_64>;

    namespace plugins
    {
        namespace shadowgraphy
        {
            namespace po = boost::program_options;
            class Shadowgraphy : public plugins::multi::IInstance
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
                        return std::shared_ptr<IInstance>(new Shadowgraphy(help, id, cellDescription));
                    }

                    //! periodicity of computing the particle energy
                    plugins::multi::Option<int> optionStart
                        = {"start", "step to start plugin [for each n-th step]", 0};
                    plugins::multi::Option<int> optionDuration
                        = {"duration", "number of steps used to aggregate fields: 0 is disabling the plugin", 0};
                    plugins::multi::Option<std::string> optionFileName
                        = {"file", "file name to store slices in: ", "shadowgram"};
                    plugins::multi::Option<std::string> optionFileExtention
                        = {"ext",
                           "openPMD filename extension. This controls the"
                           "backend picked by the openPMD API. Available extensions: ["
                               + openPMD::printAvailableExtensions() + "]",
                           openPMD::getDefaultExtension().c_str()};
                    plugins::multi::Option<float_X> optionSlicePoint
                        = {"slicePoint", "slice point in the direction 0.0 <= x < 1.0", 0.5};
                    plugins::multi::Option<float_X> optionFocusPosition
                        = {"focusPos", "focus position relative to slice point [in meter]", 0.0};
                    plugins::multi::Option<bool> optionFourierOutput
                        = {"fourierOutput",
                           "optional output: E and B fields in (x, y, omega) Fourier space, 1==enabled",
                           0};
                    plugins::multi::Option<bool> optionIntermediateOutput
                        = {"intermediateOutput",
                           "optional output: E and B fields in (kx, ky, omega) Fourier space, 1==enabled",
                           0};


                    ///! method used by plugin controller to get --help description
                    void registerHelp(
                        boost::program_options::options_description& desc,
                        std::string const& masterPrefix = std::string{}) override
                    {
                        optionStart.registerHelp(desc, masterPrefix + prefix);
                        optionFileName.registerHelp(desc, masterPrefix + prefix);
                        optionFileExtention.registerHelp(desc, masterPrefix + prefix);
                        optionSlicePoint.registerHelp(desc, masterPrefix + prefix);
                        optionFocusPosition.registerHelp(desc, masterPrefix + prefix);
                        optionDuration.registerHelp(desc, masterPrefix + prefix);
                        optionFourierOutput.registerHelp(desc, masterPrefix + prefix);
                        optionIntermediateOutput.registerHelp(desc, masterPrefix + prefix);
                    }

                    void expandHelp(
                        boost::program_options::options_description& desc,
                        std::string const& masterPrefix = std::string{}) override
                    {
                    }


                    void validateOptions() override
                    {
                        for(int i = 0; i < optionStart.size(); ++i)
                        {
                            if(optionStart.get(i) < 0)
                                throw std::runtime_error(
                                    name + ": plugin must start after the simulation was started");
                        }
                        for(int i = 0; i < optionDuration.size(); ++i)
                        {
                            if(optionDuration.get(i) <= 0)
                                throw std::runtime_error(name + ": plugin duration must be larger than 0");
                        }
                        for(int i = 0; i < optionSlicePoint.size(); ++i)
                        {
                            if((optionSlicePoint.get(i) < 0) || (optionSlicePoint.get(i) > 1.0))
                                throw std::runtime_error(name + ": the plugin slice point must be between 0 and 1");
                        }
                    }

                    size_t getNumPlugins() const override
                    {
                        return optionDuration.size();
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

                    std::string const name = "Shadowgraphy";
                    //! short description of the plugin
                    std::string const description
                        = "Calculate the energy density of a laser by integrating the Pointing vectors in a spatially "
                          "fixed slice over a given time interval.";
                    //! prefix used for command line arguments
                    std::string const prefix = "shadowgraphy";
                };

            private:
                MappingDesc* m_cellDescription = nullptr;

                // do not change the plane, code is only supporting a plane in z direction
                int plane = 2;

                bool isIntegrating = false;
                int startTime = 0;
                // duration adjusted to be a multiple of params::tRes
                int adjustedDuration = 0;

                int localPlaneIdx = -1;

                std::unique_ptr<shadowgraphy::Helper> helper;
                std::unique_ptr<pmacc::mpi::GatherSlice> gather;

                std::shared_ptr<Help> m_help;
                size_t m_id;

            public:
                Shadowgraphy(
                    std::shared_ptr<plugins::multi::IHelp>& help,
                    size_t const id,
                    MappingDesc* cellDescription)
                    : m_cellDescription(cellDescription)
                    , m_help(std::static_pointer_cast<Help>(help))
                    , m_id(id)
                {
                    static_assert(simDim == DIM3, "Shadowgraphy-plugin requires 3D simulations.");
                    init();
                }

                void init()
                {
                    auto duration = m_help->optionDuration.get(m_id);
                    // adjust to be a multiple of params::tRes
                    adjustedDuration = (duration / params::tRes) * params::tRes;
                    auto startStep = m_help->optionStart.get(m_id);
                    auto slicePoint = m_help->optionSlicePoint.get(m_id);
                    // called when plugin is loaded, command line flags are available here
                    // set notification period for our plugin at the PluginConnector

                    // The plugin integrates the Pointing vectors over time and must thus be called every
                    // tRes-th time-step of the simulation until the integration is done
                    int lastStep = startStep + adjustedDuration;

                    std::string internalNotifyPeriod = std::to_string(startStep) + ":" + std::to_string(lastStep) + ":"
                        + std::to_string(params::tRes);

                    Environment<>::get().PluginConnector().setNotificationPeriod(this, internalNotifyPeriod);

                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    auto globalDomain = subGrid.getGlobalDomain();
                    auto globalPlaneExtent = globalDomain.size[plane];
                    auto localDomain = subGrid.getLocalDomain();

                    auto globalPlaneIdx = globalPlaneExtent * slicePoint;

                    auto isPlaneInLocalDomain = globalPlaneIdx >= localDomain.offset[plane]
                        && globalPlaneIdx < localDomain.offset[plane] + localDomain.size[plane];
                    if(isPlaneInLocalDomain)
                        localPlaneIdx = globalPlaneIdx - localDomain.offset[plane];


                    gather = std::make_unique<pmacc::mpi::GatherSlice>();
                    gather->participate(isPlaneInLocalDomain);
                }

                void restart(uint32_t restartStep, std::string const& restartDirectory) override
                {
                    ///@todo please implement
                }

                void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
                {
                    ///@todo please implement
                }

                //! must be implemented by the user
                static std::shared_ptr<plugins::multi::IHelp> getHelp()
                {
                    return std::shared_ptr<plugins::multi::IHelp>(new Help{});
                }

                //! Implementation of base class function.
                void notify(uint32_t currentStep) override
                {
                    // skip notify, slice is not intersecting the local domain
                    if(!gather->isParticipating())
                        return;
                    /* notification callback for simulation step currentStep
                     * called every notifyPeriod steps */
                    // First time the plugin is called:
                    if(isIntegrating == false)
                    {
                        startTime = currentStep;

                        if(gather->isMaster() && helper == nullptr)
                        {
                            auto slicePoint = m_help->optionSlicePoint.get(m_id);
                            helper = std::make_unique<Helper>(
                                currentStep,
                                slicePoint,
                                m_help->optionFocusPosition.get(m_id),
                                adjustedDuration,
                                m_help->optionFourierOutput.get(m_id));
                        }
                        isIntegrating = true;
                    }

                    // Convert currentStep (simulation time-step) into localStep for time domain DFT
                    int localStep = (currentStep - startTime) / params::tRes;

                    bool const dumpFinalData = localStep == (adjustedDuration / params::tRes);
                    if(!dumpFinalData)
                    {
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto inputFieldBufferE = dc.get<FieldE>(FieldE::getName());
                        inputFieldBufferE->synchronize();
                        auto sliceBufferE
                            = getGlobalSlice<shadowgraphy::Helper::FieldType::E>(inputFieldBufferE, localPlaneIdx);
                        if(gather->isMaster())
                        {
                            helper->storeField<shadowgraphy::Helper::FieldType::E>(
                                localStep,
                                currentStep,
                                sliceBufferE);
                        }

                        auto inputFieldBufferB = dc.get<FieldB>(FieldB::getName());
                        inputFieldBufferB->synchronize();
                        auto sliceBufferB
                            = getGlobalSlice<shadowgraphy::Helper::FieldType::B>(inputFieldBufferB, localPlaneIdx);
                        if(gather->isMaster())
                        {
                            helper->storeField<shadowgraphy::Helper::FieldType::B>(
                                localStep,
                                currentStep,
                                sliceBufferB);
                        }

                        if(gather->isMaster())
                        {
                            helper->computeDFT(localStep);
                        }
                    }
                    else
                    {
                        if(gather->isMaster())
                        {
                            // delete helper and free all memory
                            helper.reset(nullptr);
                        }
                        isIntegrating = false;
                    }
                }

            private:
                /** Create and store the global slice out of local data.
                 *
                 * Create the slice of the local field. The field values will be interpolated to the origin of the
                 * cell. Gather the local field data into a single global field on the gather master.
                 *
                 * @tparam T_fieldType
                 * @tparam T_Buffer
                 * @param inputFieldBuffer
                 * @param cellIdxZ
                 * @return Buffer with gathered global slice. (only gather master buffer contains data)
                 */
                template<typename shadowgraphy::Helper::FieldType T_fieldType, typename T_Buffer>
                auto getGlobalSlice(std::shared_ptr<T_Buffer> inputFieldBuffer, int cellIdxZ) const
                    -> std::shared_ptr<HostBuffer<float2_X, DIM2>>
                {
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    auto globalDomain = subGrid.getGlobalDomain();
                    auto globalPlaneExtent = globalDomain.size[plane];

                    auto localDomainOffset = subGrid.getLocalDomain().offset.shrink<DIM2>(0);
                    auto globalDomainSliceSize = subGrid.getGlobalDomain().size.shrink<DIM2>(0);

                    auto fieldSlice = createSlice<T_fieldType>(inputFieldBuffer, cellIdxZ);
                    return gather->gatherSlice(*fieldSlice, globalDomainSliceSize, localDomainOffset);
                }

                template<typename shadowgraphy::Helper::FieldType T_fieldType, typename T_FieldBuffer>
                auto createSlice(std::shared_ptr<T_FieldBuffer> inputFieldBuffer, int sliceCellZ) const
                {
                    auto bufferGridLayout = inputFieldBuffer->getGridLayout();
                    DataSpace<DIM2> localSliceSize = bufferGridLayout.sizeWithoutGuardND().template shrink<DIM2>(0);

                    // skip guard cells
                    auto inputFieldBox = inputFieldBuffer->getHostDataBox().shift(bufferGridLayout.guardSizeND());

                    auto sliceBuffer = std::make_shared<HostBuffer<float2_X, DIM2>>(localSliceSize);
                    auto sliceBox = sliceBuffer->getDataBox();

                    for(int y = 0; y < localSliceSize.y(); ++y)
                        for(int x = 0; x < localSliceSize.x(); ++x)
                        {
                            DataSpace<DIM2> idx(x, y);
                            DataSpace<DIM3> srcIdx(idx.x(), idx.y(), sliceCellZ);
                            sliceBox(idx) = helper->cross<T_fieldType>(inputFieldBox.shift(srcIdx));
                        }

                    return sliceBuffer;
                }
            };

        } // namespace shadowgraphy
    } // namespace plugins
} // namespace picongpu
