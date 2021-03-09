/*
 * Copyright 2013-2021 Alexander Matthes,
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

// Needs to be the very first
#include <boost/fusion/include/mpl.hpp>

#include "picongpu/plugins/ILightweightPlugin.hpp"
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/static_assert.hpp>

#define ISAAC_IDX_TYPE cupla::IdxType
#include <isaac.hpp>

#include <boost/fusion/container/list.hpp>
#include <boost/fusion/include/list.hpp>
#include <boost/fusion/container/list/list_fwd.hpp>
#include <boost/fusion/include/list_fwd.hpp>
#include <boost/fusion/include/as_list.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/transform.hpp>
#include <limits>

namespace picongpu
{
    namespace isaacP
    {
        using namespace pmacc;
        using namespace ::isaac;

        ISAAC_NO_HOST_DEVICE_WARNING
        template<typename FieldType>
        class TFieldSource
        {
        public:
            static const size_t feature_dim = 3;
            static const bool has_guard = bmpl::not_<boost::is_same<FieldType, FieldJ>>::value;
            static const bool persistent = bmpl::not_<boost::is_same<FieldType, FieldJ>>::value;
            typename FieldType::DataBoxType shifted;
            MappingDesc* cellDescription;
            bool movingWindow;
            TFieldSource() : cellDescription(nullptr), movingWindow(false)
            {
            }

            void init(MappingDesc* cellDescription, bool movingWindow)
            {
                this->cellDescription = cellDescription;
                this->movingWindow = movingWindow;
            }

            static std::string getName()
            {
                return FieldType::getName() + std::string(" field");
            }

            void update(bool enabled, void* pointer)
            {
                if(enabled)
                {
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    DataConnector& dc = Environment<simDim>::get().DataConnector();
                    auto pField = dc.get<FieldType>(FieldType::getName(), true);
                    DataSpace<simDim> guarding = SuperCellSize::toRT() * cellDescription->getGuardingSuperCells();
                    if(movingWindow)
                    {
                        GridController<simDim>& gc = Environment<simDim>::get().GridController();
                        if(gc.getPosition()[1] == 0) // first gpu
                        {
                            uint32_t* currentStep = (uint32_t*) pointer;
                            Window window(MovingWindow::getInstance().getWindow(*currentStep));
                            guarding += subGrid.getLocalDomain().size - window.localDimensions.size;
                        }
                    }
                    typename FieldType::DataBoxType dataBox = pField->getDeviceDataBox();
                    shifted = dataBox.shift(guarding);
                    dc.releaseData(FieldType::getName());
                    /* avoid deadlock between not finished pmacc tasks and potential blocking operations
                     * within ISAAC
                     */
                    __getTransactionEvent().waitForFinished();
                }
            }

            ISAAC_NO_HOST_DEVICE_WARNING
            ISAAC_HOST_DEVICE_INLINE isaac_float_dim<feature_dim> operator[](const isaac_int3& nIndex) const
            {
                auto value = shifted[nIndex.z][nIndex.y][nIndex.x];
                return isaac_float_dim<feature_dim>(value.x(), value.y(), value.z());
            }
        };

        ISAAC_NO_HOST_DEVICE_WARNING
        template<typename FrameSolver, typename ParticleType>
        class TFieldSource<FieldTmpOperation<FrameSolver, ParticleType>>
        {
        public:
            static const size_t feature_dim = 1;
            static const bool has_guard = false;
            static const bool persistent = false;
            typename FieldTmp::DataBoxType shifted;
            MappingDesc* cellDescription;
            bool movingWindow;

            TFieldSource() : cellDescription(nullptr), movingWindow(false)
            {
            }

            void init(MappingDesc* cellDescription, bool movingWindow)
            {
                this->cellDescription = cellDescription;
                this->movingWindow = movingWindow;
            }

            static std::string getName()
            {
                return ParticleType::FrameType::getName() + std::string(" ") + FrameSolver().getName();
            }

            void update(bool enabled, void* pointer)
            {
                if(enabled)
                {
                    uint32_t* currentStep = (uint32_t*) pointer;
                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    DataConnector& dc = Environment<simDim>::get().DataConnector();

                    PMACC_CASSERT_MSG(_please_allocate_at_least_one_FieldTmp_in_memory_param, fieldTmpNumSlots > 0);
                    auto fieldTmp = dc.get<FieldTmp>(FieldTmp::getUniqueId(0), true);
                    auto particles = dc.get<ParticleType>(ParticleType::FrameType::getName(), true);

                    fieldTmp->getGridBuffer().getDeviceBuffer().setValue(FieldTmp::ValueType(0.0));
                    fieldTmp->template computeValue<CORE + BORDER, FrameSolver>(*particles, *currentStep);
                    EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());

                    __setTransactionEvent(fieldTmpEvent);
                    __getTransactionEvent().waitForFinished();

                    dc.releaseData(ParticleType::FrameType::getName());

                    DataSpace<simDim> guarding = SuperCellSize::toRT() * cellDescription->getGuardingSuperCells();
                    if(movingWindow)
                    {
                        GridController<simDim>& gc = Environment<simDim>::get().GridController();
                        if(gc.getPosition()[1] == 0) // first gpu
                        {
                            Window window(MovingWindow::getInstance().getWindow(*currentStep));
                            guarding += subGrid.getLocalDomain().size - window.localDimensions.size;
                        }
                    }
                    typename FieldTmp::DataBoxType dataBox = fieldTmp->getDeviceDataBox();
                    shifted = dataBox.shift(guarding);
                    dc.releaseData(FieldTmp::getUniqueId(0));
                }
            }

            ISAAC_NO_HOST_DEVICE_WARNING
            ISAAC_HOST_DEVICE_INLINE isaac_float_dim<feature_dim> operator[](const isaac_int3& nIndex) const
            {
                auto value = shifted[nIndex.z][nIndex.y][nIndex.x];
                return isaac_float_dim<feature_dim>(value.x());
            }
        };


        template<size_t feature_dim, typename ParticlesBoxType>
        class ParticleIterator
        {
        public:
            using FramePtr = typename ParticlesBoxType::FramePtr;
            // size of the particle list
            size_t size;

            ISAAC_NO_HOST_DEVICE_WARNING
            ISAAC_HOST_DEVICE_INLINE ParticleIterator(
                size_t size,
                ParticlesBoxType pb,
                FramePtr firstFrame,
                int frameSize)
                : size(size)
                , pb(pb)
                , frame(firstFrame)
                , frameSize(frameSize)
                , i(0)
            {
            }

            ISAAC_HOST_DEVICE_INLINE void next()
            {
                // iterate particles look for next frame
                ++i;
                if(i >= frameSize)
                {
                    frame = pb.getNextFrame(frame);
                    i = 0;
                }
            }

            // returns current particle position
            ISAAC_HOST_DEVICE_INLINE isaac_float3 getPosition() const
            {
                auto const particle = frame[i];

                // storage number in the actual frame
                const auto frameCellNr = particle[localCellIdx_];

                // offset in the actual superCell = cell offset in the supercell
                const DataSpace<simDim> frameCellOffset(
                    DataSpaceOperations<simDim>::template map<MappingDesc::SuperCellSize>(frameCellNr));

                // added offsets
                float3_X const absoluteOffset(particle[position_] + float3_X(frameCellOffset));

                // calculate scaled position
                isaac_float3 const pos(
                    absoluteOffset.x() * (1._X / float_X(MappingDesc::SuperCellSize::x::value)),
                    absoluteOffset.y() * (1._X / float_X(MappingDesc::SuperCellSize::y::value)),
                    absoluteOffset.z() * (1._X / float_X(MappingDesc::SuperCellSize::z::value)));

                return pos;
            }

            // returns particle momentum as color attribute
            ISAAC_HOST_DEVICE_INLINE isaac_float_dim<feature_dim> getAttribute() const
            {
                auto const particle = frame[i];
                float3_X const mom = particle[momentum_];
                return isaac_float_dim<feature_dim>(mom[0], mom[1], mom[2]);
            }


            // returns constant radius
            ISAAC_HOST_DEVICE_INLINE isaac_float getRadius() const
            {
                return 0.2f;
            }


        private:
            ParticlesBoxType pb;
            FramePtr frame;
            int i;
            int frameSize;
        };


        ISAAC_NO_HOST_DEVICE_WARNING
        template<typename ParticlesType>
        class ParticleSource
        {
            using ParticlesBoxType = typename ParticlesType::ParticlesBoxType;
            using FramePtr = typename ParticlesBoxType::FramePtr;
            using FrameType = typename ParticlesBoxType::FrameType;

        public:
            static const size_t feature_dim = 3;
            bool movingWindow;
            DataSpace<simDim> guarding;
            ISAAC_NO_HOST_DEVICE_WARNING
            ParticleSource()
            {
            }

            ISAAC_HOST_INLINE static std::string getName()
            {
                return ParticlesType::FrameType::getName() + std::string(" particle");
            }

            pmacc::memory::Array<ParticlesBoxType, 1> pb;

            void init(bool movingWindow)
            {
                this->movingWindow = movingWindow;
            }

            void update(bool enabled, void* pointer)
            {
                // update movingWindow cells
                if(enabled)
                {
                    uint32_t* currentStep = (uint32_t*) pointer;
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto particles = dc.get<ParticlesType>(ParticlesType::FrameType::getName(), true);
                    pb[0] = particles->getDeviceParticlesBox();

                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
                    guarding = GuardSize::toRT();
                    if(movingWindow)
                    {
                        GridController<simDim>& gc = Environment<simDim>::get().GridController();
                        if(gc.getPosition()[1] == 0) // first gpu
                        {
                            Window window(MovingWindow::getInstance().getWindow(*currentStep));
                            for(uint32_t i = 0; i < simDim; i++)
                                guarding[i] += int(math::ceil(
                                    (subGrid.getLocalDomain().size[i] - window.localDimensions.size[i])
                                    / (float) MappingDesc::SuperCellSize::toRT()[i]));
                        }
                    }
                    dc.releaseData(ParticlesType::FrameType::getName());
                }
            }

            // returns particleIterator with correct feature_dim and cell specific particlebox
            ISAAC_NO_HOST_DEVICE_WARNING
            ISAAC_HOST_DEVICE_INLINE ParticleIterator<feature_dim, ParticlesBoxType> getIterator(
                const isaac_uint3& local_grid_coord) const
            {
                constexpr uint32_t frameSize = pmacc::math::CT::volume<typename FrameType::SuperCellSize>::type::value;
                DataSpace<simDim> const superCellIdx(
                    local_grid_coord.x + guarding[0],
                    local_grid_coord.y + guarding[1],
                    local_grid_coord.z + guarding[2]);
                const auto& superCell = pb[0].getSuperCell(superCellIdx);
                size_t size = superCell.getNumParticles();
                FramePtr currentFrame = pb[0].getFirstFrame(superCellIdx);
                return ParticleIterator<feature_dim, ParticlesBoxType>(size, pb[0], currentFrame, frameSize);
            }
        };

        template<typename T>
        struct Transformoperator
        {
            typedef TFieldSource<T> type;
        };
        template<typename T>
        struct ParticleTransformoperator
        {
            typedef ParticleSource<T> type;
        };

        struct SourceInitIterator
        {
            template<typename TSource, typename TCellDescription, typename TMovingWindow>
            void operator()(const int I, TSource& s, TCellDescription& c, TMovingWindow& w) const
            {
                s.init(c, w);
            }
        };

        struct ParticleSourceInitIterator
        {
            template<typename TParticleSource, typename TMovingWindow>
            void operator()(const int I, TParticleSource& s, TMovingWindow& w) const
            {
                s.init(w);
            }
        };


        class IsaacPlugin : public ILightweightPlugin
        {
        public:
            static const ISAAC_IDX_TYPE textureDim = 1024;
            using SourceList = bmpl::
                transform<boost::fusion::result_of::as_list<Fields_Seq>::type, Transformoperator<bmpl::_1>>::type;
            // create compile time particle list
            using ParticleList = bmpl::transform<
                boost::fusion::result_of::as_list<Particle_Seq>::type,
                ParticleTransformoperator<bmpl::_1>>::type;
            using VisualizationType = IsaacVisualization<
                cupla::AccHost,
                cupla::Acc,
                cupla::AccStream,
                cupla::KernelDim,
                ParticleList,
                SourceList,
                textureDim,
#if(ISAAC_STEREO == 0)
                isaac::DefaultController,
                isaac::DefaultCompositor
#else
                isaac::StereoController,
#    if(ISAAC_STEREO == 1)
                isaac::StereoCompositorSideBySide<isaac::StereoController>
#    else
                isaac::StereoCompositorAnaglyph<isaac::StereoController, 0x000000FF, 0x00FFFF00>
#    endif
#endif
                >;
            VisualizationType* visualization;

            IsaacPlugin()
                : visualization(nullptr)
                , cellDescription(nullptr)
                , movingWindow(false)
                , render_interval(1)
                , step(0)
                , drawing_time(0)
                , cell_count(0)
                , particle_count(0)
                , last_notify(0)
            {
                Environment<>::get().PluginConnector().registerPlugin(this);
            }

            std::string pluginGetName() const
            {
                return "IsaacPlugin";
            }

            void notify(uint32_t currentStep)
            {
                uint64_t simulation_time = visualization->getTicksUs() - last_notify;
                step++;
                if(step >= render_interval)
                {
                    step = 0;
                    bool pause = false;
                    do
                    {
                        // update of the position for moving window simulations
                        if(movingWindow)
                        {
                            Window window(MovingWindow::getInstance().getWindow(currentStep));
                            isaac_size3 position;
                            isaac_size3 local_size;
                            isaac_size3 particle_size;

                            for(ISAAC_IDX_TYPE i = 0; i < 3; ++i)
                            {
                                position[i] = window.localDimensions.offset[i];
                                local_size[i] = window.localDimensions.size[i];
                                particle_size[i]
                                    = window.localDimensions.size[i] / MappingDesc::SuperCellSize::toRT()[i];
                            }
                            visualization->updatePosition(position);
                            visualization->updateLocalSize(local_size);
                            visualization->updateLocalParticleSize(particle_size);
                            visualization->updateBounding();
                        }
                        if(rank == 0 && visualization->kernel_time)
                        {
                            json_object_set_new(
                                visualization->getJsonMetaRoot(),
                                "time step",
                                json_integer(currentStep));
                            json_object_set_new(
                                visualization->getJsonMetaRoot(),
                                "drawing_time",
                                json_integer(drawing_time));
                            json_object_set_new(
                                visualization->getJsonMetaRoot(),
                                "simulation_time",
                                json_integer(simulation_time));
                            simulation_time = 0;
                            json_object_set_new(
                                visualization->getJsonMetaRoot(),
                                "cell count",
                                json_integer(cell_count));
                            json_object_set_new(
                                visualization->getJsonMetaRoot(),
                                "particle count",
                                json_integer(particle_count));
                        }
                        uint64_t start = visualization->getTicksUs();
                        json_t* meta = visualization->doVisualization(META_MASTER, &currentStep, !pause);
                        drawing_time = visualization->getTicksUs() - start;
                        json_t* json_pause = nullptr;
                        if(meta && (json_pause = json_object_get(meta, "pause")) && json_boolean_value(json_pause))
                            pause = !pause;
                        if(meta && json_integer_value(json_object_get(meta, "exit")))
                            exit(1);
                        json_t* js;
                        if(meta && (js = json_object_get(meta, "interval")))
                        {
                            render_interval = math::max(int(1), int(json_integer_value(js)));
                            // Feedback for other clients than the changing one
                            if(rank == 0)
                                json_object_set_new(
                                    visualization->getJsonMetaRoot(),
                                    "interval",
                                    json_integer(render_interval));
                        }
                        json_decref(meta);
                        if(direct_pause)
                        {
                            pause = true;
                            direct_pause = false;
                        }
                    } while(pause);
                }
                last_notify = visualization->getTicksUs();
            }

            void pluginRegisterHelp(po::options_description& desc)
            {
                /* register command line parameters for your plugin */
                desc.add_options()(
                    "isaac.period",
                    po::value<std::string>(&notifyPeriod),
                    "Enable IsaacPlugin [for each n-th step].")(
                    "isaac.name",
                    po::value<std::string>(&name)->default_value("default"),
                    "The name of the simulation. Default is \"default\".")(
                    "isaac.url",
                    po::value<std::string>(&url)->default_value("localhost"),
                    "The url of the isaac server to connect to. Default is \"localhost\".")(
                    "isaac.port",
                    po::value<uint16_t>(&port)->default_value(2460),
                    "The port of the isaac server to connect to. Default is 2460.")(
                    "isaac.width",
                    po::value<uint32_t>(&width)->default_value(1024),
                    "The width per isaac framebuffer. Default is 1024.")(
                    "isaac.height",
                    po::value<uint32_t>(&height)->default_value(768),
                    "The height per isaac framebuffer. Default is 768.")(
                    "isaac.directPause",
                    po::value<bool>(&direct_pause)->default_value(false),
                    "Direct pausing after starting simulation. Default is false.")(
                    "isaac.quality",
                    po::value<uint32_t>(&jpeg_quality)->default_value(90),
                    "JPEG quality. Default is 90.")(
                    "isaac.reconnect",
                    po::value<bool>(&reconnect)->default_value(true),
                    "Trying to reconnect every time an image is rendered if the connection is lost or could never "
                    "established at all.");
            }

            void setMappingDescription(MappingDesc* cellDescription)
            {
                this->cellDescription = cellDescription;
            }

        private:
            MappingDesc* cellDescription;
            std::string notifyPeriod;
            std::string url;
            std::string name;
            uint16_t port;
            uint32_t count;
            uint32_t width;
            uint32_t height;
            uint32_t jpeg_quality;
            int rank;
            int numProc;
            bool movingWindow;
            ParticleList particleSources;
            SourceList sources;
            /** render interval within the notify period
             *
             * render each n-th time step within an interval defined by notifyPeriod
             */
            uint32_t render_interval;
            uint32_t step;
            int drawing_time;
            bool direct_pause;
            int cell_count;
            int particle_count;
            uint64_t last_notify;
            bool reconnect;

            void pluginLoad()
            {
                if(!notifyPeriod.empty())
                {
                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                    MPI_Comm_size(MPI_COMM_WORLD, &numProc);
                    if(MovingWindow::getInstance().isEnabled())
                        movingWindow = true;
                    isaac_float minCellSize = math::min(cellSize[0], math::min(cellSize[1], cellSize[2]));
                    isaac_float3 cellSizeFactor(
                        cellSize[0] / minCellSize,
                        cellSize[1] / minCellSize,
                        cellSize[2] / minCellSize);

                    const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

                    isaac_size2 framebuffer_size = {cupla::IdxType(width), cupla::IdxType(height)};

                    isaac_for_each_params(sources, SourceInitIterator(), cellDescription, movingWindow);
                    isaac_for_each_params(particleSources, ParticleSourceInitIterator(), movingWindow);

                    isaac_size3 global_size;
                    isaac_size3 local_size;
                    isaac_size3 particle_size;
                    isaac_size3 position;
                    for(ISAAC_IDX_TYPE i = 0; i < 3; ++i)
                    {
                        global_size[i] = MovingWindow::getInstance().getWindow(0).globalDimensions.size[i];
                        local_size[i] = subGrid.getLocalDomain().size[i];
                        particle_size[i] = subGrid.getLocalDomain().size[i] / SuperCellSize::toRT()[i];
                        position[i] = subGrid.getLocalDomain().offset[i];
                    }
                    visualization = new VisualizationType(
                        cupla::manager::Device<cupla::AccHost>::get().current(),
                        cupla::manager::Device<cupla::AccDev>::get().current(),
                        cupla::manager::Stream<cupla::AccDev, cupla::AccStream>::get().stream(),
                        name,
                        0,
                        url,
                        port,
                        framebuffer_size,
                        global_size,
                        local_size,
                        particle_size,
                        position,
                        particleSources,
                        sources,
                        cellSizeFactor);
                    visualization->setJpegQuality(jpeg_quality);
                    // Defining the later periodicly sent meta data
                    if(rank == 0)
                    {
                        json_object_set_new(visualization->getJsonMetaRoot(), "time step", json_string("Time step"));
                        json_object_set_new(
                            visualization->getJsonMetaRoot(),
                            "drawing time",
                            json_string("Drawing time in us"));
                        json_object_set_new(
                            visualization->getJsonMetaRoot(),
                            "simulation time",
                            json_string("Simulation time in us"));
                        json_object_set_new(
                            visualization->getJsonMetaRoot(),
                            "cell count",
                            json_string("Total numbers of cells"));
                        json_object_set_new(
                            visualization->getJsonMetaRoot(),
                            "particle count",
                            json_string("Total numbers of particles"));
                    }
                    CommunicatorSetting communicatorBehaviour = reconnect ? RetryEverySend : ReturnAtError;
                    if(visualization->init(communicatorBehaviour) != 0)
                    {
                        if(rank == 0)
                            log<picLog::INPUT_OUTPUT>("ISAAC Init failed, disable plugin");
                        notifyPeriod = "";
                    }
                    else
                    {
                        const int localNrOfCells
                            = cellDescription->getGridLayout().getDataSpaceWithoutGuarding().productOfComponents();
                        cell_count = localNrOfCells * numProc;
                        particle_count = localNrOfCells * particles::TYPICAL_PARTICLES_PER_CELL
                            * (bmpl::size<VectorAllSpecies>::type::value) * numProc;
                        last_notify = visualization->getTicksUs();
                        if(rank == 0)
                            log<picLog::INPUT_OUTPUT>("ISAAC Init succeded");
                    }
                }
                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
            }

            void pluginUnload()
            {
                if(!notifyPeriod.empty())
                {
                    delete visualization;
                    visualization = nullptr;
                    if(rank == 0)
                        log<picLog::INPUT_OUTPUT>("ISAAC finished");
                }
            }
        };

    } // namespace isaacP
} // namespace picongpu
