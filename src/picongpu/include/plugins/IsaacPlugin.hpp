/**
* Copyright 2013-2016 Alexander Matthes,
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

#include "plugins/ILightweightPlugin.hpp"
#include "dataManagement/DataConnector.hpp"
#include <isaac.hpp>
#include <boost/fusion/include/at.hpp>

#if ENABLE_IONS == 1
    #define ENABLE_IONS_SAVED 1
#else
    #define ENABLE_IONS_SAVED 0
#endif
#undef ENABLE_IONS
#define ENABLE_IONS 0

namespace picongpu
{
namespace isaacP
{


using namespace PMacc;
using namespace ::isaac;

ISAAC_NO_HOST_DEVICE_WARNING
template <typename FieldType>
class TSource
{
    public:
        static const std::string name;
        static const size_t feature_dim = 3;
        static const bool has_guard = true;
        static const bool persistent = true;
        typename FieldType::DataBoxType shifted;
        MappingDesc *cellDescription;
        bool movingWindow;
        
        TSource() : cellDescription(NULL), movingWindow(false) {}
        TSource(MappingDesc *cellDescription, bool movingWindow) : cellDescription(cellDescription), movingWindow(movingWindow) {}

        void update(bool enabled, void* pointer)
        {
            const SubGrid<simDim>& subGrid = Environment< simDim >::get().SubGrid();
            DataConnector &dc = Environment< simDim >::get().DataConnector(); 
            FieldType * pField = &(dc.getData< FieldType > (FieldType::getName(), true));
            DataSpace< simDim > guarding = SuperCellSize::toRT() * cellDescription->getGuardingSuperCells();
            if (movingWindow)
            {
                GridController<simDim> &gc = Environment<simDim>::get().GridController();
                if (gc.getPosition()[1] == 0) //first gpu
                {
					uint32_t* currentStep = (uint32_t*)pointer;
                    Window window( MovingWindow::getInstance().getWindow( *currentStep ) );
                    guarding += subGrid.getLocalDomain().size - window.localDimensions.size;
                }
            }
            typename FieldType::DataBoxType dataBox = pField->getDeviceDataBox();
            shifted = dataBox.shift( guarding );
        }

        ISAAC_NO_HOST_DEVICE_WARNING
        ISAAC_HOST_DEVICE_INLINE isaac_float_dim<3> operator[] (const isaac_int3& nIndex) const
        {
            auto value = shifted[nIndex.z][nIndex.y][nIndex.x];
            isaac_float_dim<3> result =
            {
                isaac_float( value.x() ),
                isaac_float( value.y() ),
                isaac_float( value.z() )
            };
            return result;
        }
};

template <>
const std::string TSource<FieldE>::name = "Electric Field";

template <>
const std::string TSource<FieldB>::name = "Magnetic Field";

ISAAC_NO_HOST_DEVICE_WARNING
template <typename FieldType>
class TSource_Current
{
    public:
        static const std::string name;
        static const size_t feature_dim = 3;
        static const bool has_guard = false;
        static const bool persistent = false;
        typename FieldType::DataBoxType shifted;
        MappingDesc *cellDescription;
        bool movingWindow;
        
        TSource_Current() : cellDescription(NULL), movingWindow(false) {}
        TSource_Current(MappingDesc *cellDescription, bool movingWindow) : cellDescription(cellDescription), movingWindow(movingWindow) {}

        void update(bool enabled, void* pointer)
        {
            const SubGrid<simDim>& subGrid = Environment< simDim >::get().SubGrid();
            DataConnector &dc = Environment< simDim >::get().DataConnector(); 
            FieldType * pField = &(dc.getData< FieldType > (FieldType::getName(), true));
            DataSpace< simDim > guarding = SuperCellSize::toRT() * cellDescription->getGuardingSuperCells();
            if (movingWindow)
            {
                GridController<simDim> &gc = Environment<simDim>::get().GridController();
                if (gc.getPosition()[1] == 0) //first gpu
                {
					uint32_t* currentStep = (uint32_t*)pointer;
                    Window window(MovingWindow::getInstance().getWindow( *currentStep ));
                    guarding += subGrid.getLocalDomain().size - window.localDimensions.size;
                }
            }
            typename FieldType::DataBoxType dataBox = pField->getDeviceDataBox();
            shifted = dataBox.shift( guarding );
        }

        ISAAC_NO_HOST_DEVICE_WARNING
        ISAAC_HOST_DEVICE_INLINE isaac_float_dim<3> operator[] (const isaac_int3& nIndex) const
        {
            auto value = shifted[nIndex.z][nIndex.y][nIndex.x];
            isaac_float_dim<3> result =
            {
                isaac_float( value.x() ),
                isaac_float( value.y() ),
                isaac_float( value.z() )
            };
            return result;
        }
};

template <>
const std::string TSource_Current<FieldJ>::name = "Current Field";

ISAAC_NO_HOST_DEVICE_WARNING
template <typename ParticleType>
class PSource
{
    public:
        static const std::string name;
        static const size_t feature_dim = 1;
        static const bool has_guard = false;
        static const bool persistent = false;
        typename FieldTmp::DataBoxType shifted;
        MappingDesc *cellDescription;
        bool movingWindow;
        
        PSource() : cellDescription(NULL), movingWindow(false) {}
        PSource(MappingDesc *cellDescription, bool movingWindow) : cellDescription(cellDescription), movingWindow(movingWindow) {}

        void update(bool enabled, void* pointer)
        {
            if (enabled)
            {
                uint32_t* currentStep = (uint32_t*)pointer;
                const SubGrid<simDim>& subGrid = Environment< simDim >::get().SubGrid();
                DataConnector &dc = Environment< simDim >::get().DataConnector(); 
                FieldTmp * fieldTmp = &(dc.getData< FieldTmp > (FieldTmp::getName(), true));
                ParticleType * particles = &(dc.getData< ParticleType > ( ParticleType::FrameType::getName(), true));
                typedef typename CreateDensityOperation< ParticleType >::type::Solver FrameSolver;
                fieldTmp->getGridBuffer().getDeviceBuffer().setValue( FieldTmp::ValueType(0.0) );
                fieldTmp->computeValue < CORE + BORDER, FrameSolver > (*particles, *currentStep);
                EventTask fieldTmpEvent = fieldTmp->asyncCommunication(__getTransactionEvent());
                __setTransactionEvent(fieldTmpEvent);
                __getTransactionEvent().waitForFinished();
                DataSpace< simDim > guarding = SuperCellSize::toRT() * cellDescription->getGuardingSuperCells();
				if (movingWindow)
				{
					GridController<simDim> &gc = Environment<simDim>::get().GridController();
					if (gc.getPosition()[1] == 0) //first gpu
					{
						Window window(MovingWindow::getInstance().getWindow( *currentStep ));
						guarding += subGrid.getLocalDomain().size - window.localDimensions.size;
					}
				}
                typename FieldTmp::DataBoxType dataBox = fieldTmp->getDeviceDataBox();
                shifted = dataBox.shift( guarding );
            }
        }

        ISAAC_NO_HOST_DEVICE_WARNING
        ISAAC_HOST_DEVICE_INLINE isaac_float_dim<1> operator[] (const isaac_int3& nIndex) const
        {
            auto value = shifted[nIndex.z][nIndex.y][nIndex.x];
            isaac_float_dim<1> result = { isaac_float( value.x() ) };
            return result;
        }
};

template <>
const std::string PSource<PIC_Electrons>::name = "Electron density";

template <>
const std::string PSource<PIC_Ions>::name = "Ion density";

#if ENABLE_ELECTRONS_2 == 1
    template <>
    const std::string PSource<PIC_Electrons2>::name = "Electron density 2";
#endif

class IsaacPlugin : public ILightweightPlugin
{
public:
    typedef boost::mpl::int_< simDim > SimDim;
    typedef TSource<FieldE> ESource;
    typedef TSource<FieldB> BSource;
    typedef TSource_Current<FieldJ> JSource;
    typedef PSource<PIC_Electrons> EPSource;
    typedef PSource<PIC_Ions> IPSource;
    #if ENABLE_ELECTRONS_2 == 1
        typedef PSource<PIC_Electrons2> EPSource2;
    #endif
    static const size_t textureDim = 1024;
    using SourceList = boost::fusion::list
    <
        ESource
        ,BSource
        ,JSource
    #if (ENABLE_ELECTRONS == 1)
        ,EPSource
    #endif
    #if (ENABLE_ELECTRONS_2 == 1)
        ,EPSource2
    #endif
    #if (ENABLE_IONS == 1)
        ,IPSource
    #endif
    >;
    IsaacVisualization
    <
        SimDim,
        SourceList,
        DataSpace< simDim >,
        textureDim, float3_X,
        #if (ISAAC_STEREO == 0)
            isaac::DefaultController,
            isaac::DefaultCompositor
        #else
            isaac::StereoController,
            #if (ISAAC_STEREO == 1)
                isaac::StereoCompositorSideBySide<isaac::StereoController>
            #else
                isaac::StereoCompositorAnaglyph<isaac::StereoController,0x000000FF,0x00FFFF00>
            #endif
        #endif
    > * visualization;

    IsaacPlugin() :
        visualization(NULL),
        cellDescription(NULL),
        movingWindow(false),
        interval(0),
        step(0),
        drawing_time(0)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    std::string pluginGetName() const
    {
        return "IsaacPlugin";
    }

    void notify(uint32_t currentStep)
    {
        step++;
        if (step >= interval)
        {
            step = 0;
            bool pause = false;
            do
            {
                //update of the position for moving window simulations
                if ( movingWindow )
                {
                    Window window(MovingWindow::getInstance().getWindow( currentStep ));
                    visualization->updatePosition( window.localDimensions.offset );
                    visualization->updateLocalSize( window.localDimensions.size );
                    visualization->updateBounding();
                }
                if (rank == 0 && visualization->kernel_time)
                {
                    json_object_set_new( visualization->getJsonMetaRoot(), "time step", json_integer( currentStep ) );
                    json_object_set_new( visualization->getJsonMetaRoot(), "drawing_time" , json_integer( drawing_time ) );
                    json_object_set_new( visualization->getJsonMetaRoot(), "sorting_time" , json_integer( visualization->sorting_time ) );
                    json_object_set_new( visualization->getJsonMetaRoot(), "merge_time" , json_integer( visualization->merge_time - visualization->kernel_time - visualization->copy_time ) );
                    json_object_set_new( visualization->getJsonMetaRoot(), "kernel_time" , json_integer( visualization->kernel_time ) );
                    json_object_set_new( visualization->getJsonMetaRoot(), "copy_time" , json_integer( visualization->copy_time ) );
                    json_object_set_new( visualization->getJsonMetaRoot(), "video_send_time" , json_integer( visualization->video_send_time ) );
                    json_object_set_new( visualization->getJsonMetaRoot(), "buffer_time" , json_integer( visualization->buffer_time ) );
                    visualization->sorting_time = 0;
                    visualization->merge_time = 0;
                    visualization->kernel_time = 0;
                    visualization->copy_time = 0;
                    visualization->video_send_time = 0;
                    visualization->buffer_time = 0;
                }
                uint64_t start = visualization->getTicksUs(); 
                json_t* meta = visualization->doVisualization(META_MASTER, &currentStep, !pause);
                drawing_time = visualization->getTicksUs() - start;
                json_t* json_pause = NULL;
                if ( meta && (json_pause = json_object_get(meta, "pause")) && json_boolean_value( json_pause ) )
                    pause = !pause;
                if ( meta && json_integer_value( json_object_get(meta, "exit") ) )
                    exit(1);
                json_t* js;
                if ( meta && ( js = json_object_get(meta, "interval") ) )
                {
					interval = max( int(1), int( json_integer_value ( js ) ) );
					//Feedback for other clients than the changing one
					if (rank == 0)
						json_object_set_new( visualization->getJsonMetaRoot(), "interval", json_integer( interval ) );
                }
                json_decref( meta );
                if (direct_pause)
                {
                    pause = true;
                    direct_pause = false;
                }
            }
            while (pause);
        }
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        /* register command line parameters for your plugin */
        desc.add_options()
            ("isaac.period", po::value< uint32_t > (&notifyPeriod)->default_value(0),
             "Enable IsaacPlugin [for each n-th step].")
            ("isaac.name", po::value< std::string > (&name)->default_value("default"),
             "The name of the simulation. Default is \"default\".")
            ("isaac.url", po::value< std::string > (&url)->default_value("localhost"),
             "The url of the isaac server to connect to. Default is \"localhost\".")
            ("isaac.port", po::value< uint16_t > (&port)->default_value(2460),
             "The port of the isaac server to connect to. Default is 2460.")
            ("isaac.width", po::value< uint32_t > (&width)->default_value(1024),
             "The width per isaac framebuffer. Default is 1024.")
            ("isaac.height", po::value< uint32_t > (&height)->default_value(768),
             "The height per isaac framebuffer. Default is 768.")
            ("isaac.direct_pause", po::value< bool > (&direct_pause)->default_value(false),
             "Direct pausing after starting simulation. Default is false.")
            ("isaac.quality", po::value< uint32_t > (&jpeg_quality)->default_value(90),
             "JPEG quality. Default is 90.")
            ;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:
    MappingDesc *cellDescription;
    uint32_t notifyPeriod;
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
    SourceList sources;
    int interval;
    int step;
    int drawing_time;
    bool direct_pause;

    void pluginLoad()
    {
        if (notifyPeriod > 0)
        {
            interval = notifyPeriod;
            notifyPeriod = 1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &numProc);
            printf("ISAAC: Load Plugin at %i\n",rank);
            printf("%i: ISAAC activated\n",rank);
            if ( MovingWindow::getInstance().isSlidingWindowActive() )
                movingWindow = true;
            float_X minCellSize = min( cellSize[0], min( cellSize[1], cellSize[2] ) );
            float3_X cellSizeFactor = cellSize / minCellSize;
            
            const SubGrid<simDim>& subGrid = Environment< simDim >::get().SubGrid();
 
            sources = SourceList(
                ESource(cellDescription,movingWindow)
                ,BSource(cellDescription,movingWindow)
                ,JSource(cellDescription,movingWindow)
                #if (ENABLE_ELECTRONS == 1)
                    ,EPSource(cellDescription,movingWindow)
                #endif
                #if (ENABLE_ELECTRONS_2 == 1)
                    ,EPSource2(cellDescription,movingWindow)
                #endif
                #if (ENABLE_IONS == 1)
                    ,IPSource(cellDescription,movingWindow)
                #endif
            );
            isaac_size2 framebuffer_size =
            {
                size_t(width),
                size_t(height)
            };
            visualization = new IsaacVisualization
            <
                SimDim,
                SourceList,
                DataSpace< simDim >,
                textureDim, float3_X,
                #if (ISAAC_STEREO == 0)
                    isaac::DefaultController,
                    isaac::DefaultCompositor
                #else
                    isaac::StereoController,
                    #if (ISAAC_STEREO == 1)
                        isaac::StereoCompositorSideBySide<isaac::StereoController>
                    #else
                        isaac::StereoCompositorAnaglyph<isaac::StereoController,0x000000FF,0x00FFFF00>
                    #endif
                #endif
            > (
                name,
                0,
                url,
                port,
                framebuffer_size,
                subGrid.getGlobalDomain().size,
                subGrid.getLocalDomain().size,
                subGrid.getLocalDomain().offset,
                sources,
                cellSizeFactor );
            visualization->setJpegQuality(jpeg_quality);
            printf("ISAAC: Init at %i/%i\n",rank,numProc);
            if (rank == 0)
                json_object_set_new( visualization->getJsonMetaRoot(), "time step", json_string( "Time step" ) );
            if (visualization->init())
            {
                fprintf(stderr,"ISAAC: init failed at %i/%i\n",rank,numProc);
                delete visualization;
                visualization = NULL;
                notifyPeriod = 0;
            }
            printf("ISAAC: Finished init at %i\n",rank);
        }
        Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
    }

    void pluginUnload()
    {
        if (notifyPeriod > 0)
        {
            delete visualization;
            visualization = NULL;
            printf("ISAAC: Unload Plugin at %i\n",rank);
        }
    }
};

} //namespace isaac;
} //namespace picongpu;

#undef ENABLE_IONS
#define ENABLE_IONS ENABLE_IONS_SAVED
