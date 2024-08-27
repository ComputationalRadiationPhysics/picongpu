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

// required for SIMDIM definition
#include "picongpu/simulation_defines.hpp"

#if(SIMDIM == DIM3 && PIC_ENABLE_FFTW3 == 1 && ENABLE_OPENPMD == 1)

// clang-format of
#    include "picongpu/param/shadowgraphy.param"
// clang-format on

#    include "picongpu/fields/FieldB.hpp"
#    include "picongpu/fields/FieldE.hpp"
#    include "picongpu/plugins/PluginRegistry.hpp"
#    include "picongpu/plugins/common/openPMDAttributes.hpp"
#    include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#    include "picongpu/plugins/common/openPMDVersion.def"
#    include "picongpu/plugins/common/openPMDWriteMeta.hpp"
#    include "picongpu/plugins/multi/multi.hpp"
#    include "picongpu/plugins/shadowgraphy/ShadowgraphyHelper.hpp"

#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/math/Vector.hpp>
#    include <pmacc/mpi/GatherSlice.hpp>

#    include <iostream>
#    include <sstream>
#    include <string>

#    include <mpi.h>
#    include <openPMD/openPMD.hpp>
#    include <stdio.h>


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
                        for(uint32_t i = 0; i < optionStart.size(); ++i)
                        {
                            if(optionStart.get(i) < 0)
                                throw std::runtime_error(
                                    name + ": plugin must start after the simulation was started");
                        }
                        for(uint32_t i = 0; i < optionDuration.size(); ++i)
                        {
                            if(optionDuration.get(i) <= 0)
                                throw std::runtime_error(name + ": plugin duration must be larger than 0");
                        }
                        for(uint32_t i = 0; i < optionSlicePoint.size(); ++i)
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

                /** Implementation of base class function.
                 *
                 * Called every tRes'th step of the simulation after plugin start.
                 */
                void notify(uint32_t currentStep) override
                {
                    // skip notify, slice is not intersecting the local domain
                    if(!gather->isParticipating())
                        return;

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

                    bool const dumpFinalData = localStep == static_cast<int>(adjustedDuration / params::tRes);
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
                            if(m_help->optionFourierOutput.get(m_id))
                            {
                                writeFourierOutputToOpenPMDFile(currentStep);
                            }

                            helper->propagateFieldsAndCalculateShadowgram();

                            std::ostringstream filename;
                            filename << m_help->optionFileName.get(m_id) << "_" << startTime << ":" << currentStep
                                     << ".dat";

                            writeToOpenPMDFile(currentStep);

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

                //! Write shadowgram to openPMD file
                void writeToOpenPMDFile(uint32_t currentStep)
                {
                    std::stringstream filename;
                    filename << m_help->optionFileName.get(m_id) << "_%T." << m_help->optionFileExtention.get(m_id);
                    ::openPMD::Series series(filename.str(), ::openPMD::Access::CREATE);

                    ::openPMD::Extent extent
                        = {static_cast<unsigned long int>(helper->getSizeY()),
                           static_cast<unsigned long int>(helper->getSizeX())};
                    ::openPMD::Offset offset = {0, 0};
                    ::openPMD::Datatype datatype = ::openPMD::determineDatatype<float_64>();
                    ::openPMD::Dataset dataset{datatype, extent};

                    auto mesh = series.iterations[currentStep].meshes["shadowgram"];
                    mesh.setAxisLabels(std::vector<std::string>{"x", "y"});
                    mesh.setDataOrder(::openPMD::Mesh::DataOrder::F);
                    mesh.setGridUnitSI(1);
                    mesh.setGridSpacing(std::vector<double>{1.0, 1.0});
                    mesh.setAttribute<int>("duration", m_help->optionDuration.get(m_id));
                    mesh.setAttribute<float_X>("dt", UNIT_TIME * params::tRes);
                    mesh.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default

                    auto shadowgram = mesh[::openPMD::RecordComponent::SCALAR];
                    shadowgram.resetDataset(dataset);

                    // Do not delete this object before dataPtr is not required anymore
                    auto data = helper->getShadowgramBuf();
                    auto sharedDataPtr = std::shared_ptr<float_64>{data->data(), [](auto const*) {}};

                    shadowgram.storeChunk(sharedDataPtr, offset, extent);


                    ::openPMD::Mesh spatialMesh = series.iterations[currentStep].meshes["Spatial positions"];
                    spatialMesh.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default
                    spatialMesh.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    spatialMesh.setGridSpacing(std::vector<double>{1.0});
                    spatialMesh.setGridGlobalOffset(std::vector<double>{0.0});
                    spatialMesh.setGridUnitSI(1.0);
                    spatialMesh.setAxisLabels(std::vector<std::string>{"Spatial x index", "Spatial y index"});
                    spatialMesh.setUnitDimension(
                        std::map<::openPMD::UnitDimension, double>{{::openPMD::UnitDimension::L, 1.0}});

                    auto xs = std::vector<float_X>(helper->getSizeX());
                    for(int i = 0; i < helper->getSizeX(); ++i)
                    {
                        xs[i] = helper->getX(i);
                    }
                    ::openPMD::MeshRecordComponent xMRC = spatialMesh["x"];
                    xMRC.setPosition(std::vector<double>{0.0});
                    ::openPMD::Datatype datatype_x = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Extent extent_x = {1, static_cast<unsigned long int>(helper->getSizeX())};
                    ::openPMD::Dataset dataset_x = ::openPMD::Dataset(datatype_x, extent_x);
                    xMRC.resetDataset(dataset_x);

                    // Write actual data
                    ::openPMD::Offset offset_x = {0};
                    xMRC.storeChunk(xs, offset_x, extent_x);


                    auto ys = std::vector<float_X>(helper->getSizeY());
                    for(int i = 0; i < helper->getSizeY(); ++i)
                    {
                        ys[i] = helper->getY(i);
                    }

                    ::openPMD::MeshRecordComponent yMRC = spatialMesh["y"];
                    yMRC.setPosition(std::vector<double>{0.0});

                    ::openPMD::Datatype datatype_y = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Extent extent_y = {static_cast<unsigned long int>(helper->getSizeY()), 1};
                    ::openPMD::Dataset dataset_y = ::openPMD::Dataset(datatype_y, extent_y);
                    yMRC.resetDataset(dataset_y);

                    // Write actual data
                    ::openPMD::Offset offset_y = {0};
                    yMRC.storeChunk(ys, offset_y, extent_y);


                    series.iterations[currentStep].close();
                }

                //! Write Fourier output to openPMD file
                void writeFourierOutputToOpenPMDFile(uint32_t currentStep)
                {
                    std::stringstream filename;
                    filename << m_help->optionFileName.get(m_id) << "_fourierdata_%T."
                             << m_help->optionFileExtention.get(m_id);
                    ::openPMD::Series series(filename.str(), ::openPMD::Access::CREATE);

                    auto meshNeg = series.iterations[currentStep].meshes["Fourier Domain Fields - negative"];
                    meshNeg.setGeometry(::openPMD::Mesh::Geometry::cartesian);
                    meshNeg.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    meshNeg.setGridSpacing(std::vector<double>{1.0, 1.0, 1.0});
                    meshNeg.setGridGlobalOffset(
                        std::vector<double>{static_cast<double>(helper->getOmegaIndex(0)), 0.0, 0.0});
                    meshNeg.setGridUnitSI(1.0);
                    meshNeg.setAxisLabels(std::vector<std::string>{
                        "Spatial x index",
                        "Spatial y index",
                        "Fourier transform frequency index"});
                    meshNeg.setUnitDimension(std::map<::openPMD::UnitDimension, double>{
                        {::openPMD::UnitDimension::L, 1.0},
                        {::openPMD::UnitDimension::M, 1.0},
                        {::openPMD::UnitDimension::T, -3.0},
                        {::openPMD::UnitDimension::I, -1.0}});

                    // Reshape abstract MeshRecordComponent
                    ::openPMD::Datatype datatype = ::openPMD::determineDatatype<std::complex<float_64>>();
                    ::openPMD::Extent extent
                        = {static_cast<unsigned long int>(helper->getNumOmegas() / 2),
                           static_cast<unsigned long int>(helper->getSizeY()),
                           static_cast<unsigned long int>(helper->getSizeX())};
                    ::openPMD::Offset offset = {0, 0, 0};

                    // Go through all 8 different fields components: +Ex, -Ex, +Ey, -Ey, +Bx, -Bx, +By, -By
                    for(int i = 0; i < 8; i += 2)
                    {
                        std::string dir = helper->dataLabelsFieldComponent(i);
                        // Do not delete this object before dataPtr is not required anymore
                        auto data = helper->getFourierBuf(i);
                        auto sharedDataPtr
                            = std::shared_ptr<std::complex<picongpu::float_64>>{data->data(), [](auto const*) {}};
                        meshNeg[dir].setUnitSI(1.0);
                        meshNeg[dir].setPosition(std::vector<double>{0.0, 0.0, 0.0});
                        ::openPMD::Dataset dataset = ::openPMD::Dataset(datatype, extent);
                        meshNeg[dir].resetDataset(dataset);
                        meshNeg[dir].storeChunk(sharedDataPtr, offset, extent);
                        series.flush();
                    }

                    auto meshPos = series.iterations[currentStep].meshes["Fourier Domain Fields - positive"];
                    meshPos.setGeometry(::openPMD::Mesh::Geometry::cartesian);
                    meshPos.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    meshPos.setGridSpacing(std::vector<double>{1.0, 1.0, 1.0});
                    meshPos.setGridGlobalOffset(std::vector<double>{
                        static_cast<double>(helper->getOmegaIndex(helper->getNumOmegas() / 2)),
                        0.0,
                        0.0});
                    meshPos.setGridUnitSI(1.0);
                    meshPos.setAxisLabels(std::vector<std::string>{
                        "Spatial x index",
                        "Spatial y index",
                        "Fourier transform frequency index"});
                    meshPos.setUnitDimension(std::map<::openPMD::UnitDimension, double>{
                        {::openPMD::UnitDimension::L, 1.0},
                        {::openPMD::UnitDimension::M, 1.0},
                        {::openPMD::UnitDimension::T, -3.0},
                        {::openPMD::UnitDimension::I, -1.0}});
                    for(int i = 1; i < 8; i += 2)
                    {
                        std::string dir = helper->dataLabelsFieldComponent(i);
                        // do not delete this object before dataPtr is not required anymore
                        auto data = helper->getFourierBuf(i);
                        auto sharedDataPtr
                            = std::shared_ptr<std::complex<picongpu::float_64>>{data->data(), [](auto const*) {}};
                        meshPos[dir].setUnitSI(1.0);
                        meshPos[dir].setPosition(std::vector<double>{0.0, 0.0, 0.0});
                        ::openPMD::Dataset dataset = ::openPMD::Dataset(datatype, extent);
                        meshPos[dir].resetDataset(dataset);
                        meshPos[dir].storeChunk(sharedDataPtr, offset, extent);
                        series.flush();
                    }

                    auto omegas = std::vector<float_X>(helper->getNumOmegas());
                    for(int i = 0; i < helper->getNumOmegas(); ++i)
                    {
                        omegas[i] = helper->omega(helper->getOmegaIndex(i));
                    }
                    ::openPMD::Mesh meshOmega = series.iterations[currentStep].meshes["Fourier Transform Frequencies"];
                    meshOmega.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default
                    meshOmega.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    meshOmega.setGridSpacing(std::vector<double>{1.0});
                    meshOmega.setGridGlobalOffset(std::vector<double>{0.0});
                    meshOmega.setGridUnitSI(1.0);
                    meshOmega.setAxisLabels(std::vector<std::string>{"Fourier transform frequency index"});
                    meshOmega.setUnitDimension(
                        std::map<::openPMD::UnitDimension, double>{{::openPMD::UnitDimension::T, -1.0}});
                    ::openPMD::MeshRecordComponent omegaMRC = meshOmega["omegas"];
                    omegaMRC.setPosition(std::vector<double>{0.0});

                    ::openPMD::Datatype datatype_omega = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Extent extent_omega = {static_cast<unsigned long int>(helper->getNumOmegas()), 1, 1};
                    ::openPMD::Dataset dataset_omega = ::openPMD::Dataset(datatype_omega, extent_omega);
                    omegaMRC.resetDataset(dataset_omega);

                    // Write actual data
                    ::openPMD::Offset offset_omega = {0};
                    omegaMRC.storeChunk(omegas, offset_omega, extent_omega);

                    ::openPMD::Mesh spatialMesh = series.iterations[currentStep].meshes["Spatial positions"];
                    spatialMesh.setGeometry(::openPMD::Mesh::Geometry::cartesian); // set be default
                    spatialMesh.setDataOrder(::openPMD::Mesh::DataOrder::C);
                    spatialMesh.setGridSpacing(std::vector<double>{1.0});
                    spatialMesh.setGridGlobalOffset(std::vector<double>{0.0});
                    spatialMesh.setGridUnitSI(1.0);
                    spatialMesh.setAxisLabels(std::vector<std::string>{"Spatial x index", "Spatial y index", "None"});
                    spatialMesh.setUnitDimension(
                        std::map<::openPMD::UnitDimension, double>{{::openPMD::UnitDimension::L, 1.0}});

                    auto xs = std::vector<float_X>(helper->getSizeX());
                    for(int i = 0; i < helper->getSizeX(); ++i)
                    {
                        xs[i] = helper->getX(i);
                    }
                    ::openPMD::MeshRecordComponent xMRC = spatialMesh["x"];
                    xMRC.setPosition(std::vector<double>{0.0});
                    ::openPMD::Datatype datatype_x = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Extent extent_x = {1, 1, static_cast<unsigned long int>(helper->getSizeX())};
                    ::openPMD::Dataset dataset_x = ::openPMD::Dataset(datatype_x, extent_x);
                    xMRC.resetDataset(dataset_x);

                    // Write actual data
                    ::openPMD::Offset offset_x = {0};
                    xMRC.storeChunk(xs, offset_x, extent_x);


                    auto ys = std::vector<float_X>(helper->getSizeY());
                    for(int i = 0; i < helper->getSizeY(); ++i)
                    {
                        ys[i] = helper->getY(i);
                    }

                    ::openPMD::MeshRecordComponent yMRC = spatialMesh["y"];
                    yMRC.setPosition(std::vector<double>{0.0});

                    ::openPMD::Datatype datatype_y = ::openPMD::determineDatatype<float_X>();
                    ::openPMD::Extent extent_y = {1, static_cast<unsigned long int>(helper->getSizeY()), 1};
                    ::openPMD::Dataset dataset_y = ::openPMD::Dataset(datatype_y, extent_y);
                    yMRC.resetDataset(dataset_y);

                    // Write actual data
                    ::openPMD::Offset offset_y = {0};
                    yMRC.storeChunk(ys, offset_y, extent_y);


                    series.iterations[currentStep].close();
                }
            };

        } // namespace shadowgraphy
    } // namespace plugins
} // namespace picongpu

PIC_REGISTER_PLUGIN(picongpu::plugins::multi::Master<picongpu::plugins::shadowgraphy::Shadowgraphy>);

#endif
