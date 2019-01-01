/* Copyright 2013-2019 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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

#include <pthread.h>
#include <sstream>
#include <string>
#include <list>
#include <vector>
#include <regex>

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/hdf5/HDF5Writer.def"
#include "picongpu/traits/SplashToPIC.hpp"
#include "picongpu/traits/PICToSplash.hpp"
#include "picongpu/plugins/misc/misc.hpp"
#include "picongpu/plugins/multi/Option.hpp"
#include "picongpu/plugins/misc/SpeciesFilter.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/particles/filter/filter.hpp"

#include <pmacc/particles/frame_types.hpp>

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/fields/FieldTmp.hpp"
#include <pmacc/particles/particleFilter/FilterFactory.hpp>
#include <pmacc/particles/particleFilter/PositionFilter.hpp>
#include <pmacc/particles/operations/CountParticles.hpp>
#include <pmacc/particles/IdProvider.def>

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/pluginSystem/PluginConnector.hpp>
#include "picongpu/simulationControl/MovingWindow.hpp"
#include <pmacc/math/Vector.hpp>

#include "picongpu/plugins/output/IIOBackend.hpp"
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>

#include <boost/type_traits.hpp>

#include "picongpu/plugins/hdf5/WriteMeta.hpp"
#include "picongpu/plugins/hdf5/WriteFields.hpp"
#include "picongpu/plugins/hdf5/WriteSpecies.hpp"
#include "picongpu/plugins/hdf5/restart/LoadSpecies.hpp"
#include "picongpu/plugins/hdf5/restart/RestartFieldLoader.hpp"
#include "picongpu/plugins/hdf5/NDScalars.hpp"
#include "picongpu/plugins/misc/SpeciesFilter.hpp"

#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>


namespace picongpu
{

namespace hdf5
{

using namespace pmacc;

using namespace splash;

/** Writes simulation data to hdf5 files using libSplash.
 *
 * Implements the IIOBackend interface.
 */
class HDF5Writer :
    public IIOBackend
{
public:

    struct Help : public plugins::multi::IHelp
    {
        /** creates a instance of ISlave
         *
         * @tparam T_Slave type of the interface implementation (must inherit from ISlave)
         * @param help plugin defined help
         * @param id index of the plugin, range: [0;help->getNumPlugins())
         */
        std::shared_ptr< ISlave > create(
            std::shared_ptr< IHelp > & help,
            size_t const id,
            MappingDesc* cellDescription
        )
        {
            return std::shared_ptr< ISlave >(
                new HDF5Writer(
                    help,
                    id,
                    cellDescription
                )
            );
        }

        plugins::multi::Option< std::string > notifyPeriod = {
            "period",
            "enable HDF5 IO [for each n-th step]"
        };
        plugins::multi::Option< std::string > source = {
            "source",
            "data sources: ",
            "species_all, fields_all"
        };

        plugins::multi::Option< std::string > fileName = {
            "file",
            "HDF5 output filename (prefix)"
        };

        /** defines if the plugin must register itself to the PMacc plugin system
         *
         * true = the plugin is registering it self
         * false = the plugin is not registering itself (plugin is controlled by another class)
         */
        bool selfRegister = false;

        std::vector< std::string > allowedDataSources  = {
            "species_all",
            "fields_all"
        };

        template<typename T_TupleVector>
        struct CreateSpeciesFilter
        {
            using type = plugins::misc::SpeciesFilter<
                typename pmacc::math::CT::At<
                    T_TupleVector,
                    bmpl::int_<0>
                >::type,
                typename pmacc::math::CT::At<
                    T_TupleVector,
                    bmpl::int_<1>
                >::type
            >;
        };

        using AllParticlesTimesAllFilters = typename AllCombinations<
            bmpl::vector<
                FileOutputParticles,
                particles::filter::AllParticleFilters
            >
         >::type;

        using AllSpeciesFilter = typename bmpl::transform<
            AllParticlesTimesAllFilters,
            CreateSpeciesFilter< bmpl::_1 >
        >::type;

        using AllEligibleSpeciesSources = typename bmpl::copy_if<
            AllSpeciesFilter,
            plugins::misc::speciesFilter::IsEligible< bmpl::_1 >
        >::type;

        using AllFieldSources = FileOutputFields;

        ///! method used by plugin controller to get --help description
        void registerHelp(
            boost::program_options::options_description & desc,
            std::string const & masterPrefix = std::string{ }
        )
        {
            ForEach<
                AllEligibleSpeciesSources,
                plugins::misc::AppendName< bmpl::_1 >
            > getEligibleDataSourceNames;
            getEligibleDataSourceNames( forward( allowedDataSources ) );

            ForEach<
                AllFieldSources,
                plugins::misc::AppendName< bmpl::_1 >
            > appendFieldSourceNames;
            appendFieldSourceNames( forward( allowedDataSources ) );

            // string list with all possible data sources
            std::string concatenatedSourceNames = plugins::misc::concatenateToString(
                allowedDataSources,
                ", "
            );

            notifyPeriod.registerHelp(
                desc,
                masterPrefix + prefix
            );
            source.registerHelp(
                desc,
                masterPrefix + prefix,
                std::string( "[" ) + concatenatedSourceNames + "]"
            );
            fileName.registerHelp(
                desc,
                masterPrefix + prefix
            );
            selfRegister = true;

        }

        void expandHelp(
            boost::program_options::options_description & desc,
            std::string const & masterPrefix = std::string{ }
        )
        {
        }

        void validateOptions()
        {
            if( selfRegister )
            {
                if( notifyPeriod.empty() || fileName.empty() )
                    throw std::runtime_error(
                        name +
                        ": parameter period and file must be defined"
                    );

                // check if user passed data source names are valid
                for( auto const & dateSourceNames : source )
                {
                    auto vectorOfDataSourceNames = plugins::misc::splitString(
                        plugins::misc::removeSpaces( dateSourceNames )
                    );

                    for( auto const & f : vectorOfDataSourceNames )
                    {
                        if(
                            !plugins::misc::containsObject(
                                allowedDataSources,
                                f
                            )
                        )
                        {
                            throw std::runtime_error( name + ": unknown data source '" + f + "'" );
                        }
                    }
                }
            }
        }

        size_t getNumPlugins() const
        {
            if( selfRegister )
                return notifyPeriod.size();
            else
                return 1;
        }

        std::string getDescription() const
        {
            return description;
        }

        std::string getOptionPrefix() const
        {
            return prefix;
        }

        std::string getName() const
        {
            return name;
        }

        std::string const name = "HDF5Writer";
        //! short description of the plugin
        std::string const description = "dump simulation data with hdf5";
        //! prefix used for command line arguments
        std::string const prefix = "hdf5";
    };

    //! must be implemented by the user
    static std::shared_ptr< plugins::multi::IHelp > getHelp()
    {
        return std::shared_ptr< plugins::multi::IHelp >( new Help{ } );
    }

    /** constructor
     *
     * @param help instance of the class Help
     * @param id index of this plugin instance within help
     * @param cellDescription PIConGPu cell description information for kernel index mapping
     */
    HDF5Writer(
        std::shared_ptr< plugins::multi::IHelp > & help,
        size_t const id,
        MappingDesc* cellDescription
    ) :
    m_help( std::static_pointer_cast< Help >(help) ),
    m_id( id ),
    m_cellDescription( cellDescription ),
    outputDirectory("h5")
    {
        mThreadParams.cellDescription = m_cellDescription;

        GridController<simDim> &gc = Environment<simDim>::get().GridController();

        /* It is important that we never change the mpi_pos after this point
         * because we get problems with the restart.
         * Otherwise we do not know which gpu must load the ghost parts around
         * the sliding window.
         */
        mpi_pos = gc.getPosition();
        mpi_size = gc.getGpuNodes();

        splashMpiPos.set(0, 0, 0);
        splashMpiSize.set(1, 1, 1);

        for (uint32_t i = 0; i < simDim; ++i)
        {
            splashMpiPos[i] = mpi_pos[i];
            splashMpiSize[i] = mpi_size[i];
        }

        if( m_help->selfRegister )
        {
            std::string notifyPeriod = m_help->notifyPeriod.get( id );
            /* only register for notify callback when .period is set on command line */
            if(!notifyPeriod.empty())
            {
                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

                /** create notify directory */
                Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(outputDirectory);
            }
        }
    }

    virtual ~HDF5Writer()
    {
        if (mThreadParams.dataCollector)
                mThreadParams.dataCollector->finalize();

         __delete(mThreadParams.dataCollector);
    }

    void notify(uint32_t currentStep)
    {
        // notify is only allowed if the plugin is not controlled by the class Checkpoint
        assert( m_help->selfRegister );

        __getTransactionEvent().waitForFinished();

        std::string filename = m_help->fileName.get( m_id );
        /* if file name is relative, prepend with common directory */
        if( boost::filesystem::path(filename).has_root_path() )
            mThreadParams.h5Filename = filename;
        else
            mThreadParams.h5Filename = outputDirectory + "/" + filename;

        /* window selection */
        mThreadParams.window = MovingWindow::getInstance().getWindow(currentStep);
        mThreadParams.isCheckpoint = false;
        dumpData(currentStep);
    }

    virtual void restart(
        uint32_t restartStep,
        std::string const & restartDirectory
    )
    {
        /* ISlave restart interface is not needed becase IIOBackend
         * restart interface is used
         */
    }

    virtual void checkpoint(
        uint32_t currentStep,
        std::string const & checkpointDirectory
    )
    {
        /* ISlave checkpoint interface is not needed becase IIOBackend
         * checkpoint interface is used
         */
    }

    void doRestart(
        const uint32_t restartStep,
        const std::string& restartDirectory,
        const std::string& constRestartFilename,
        const uint32_t restartChunkSize
    )
    {
        // restart is only allowed if the plugin is controlled by the class Checkpoint
        assert(!m_help->selfRegister);

        // allow to modify the restart file name
        std::string restartFilename{ constRestartFilename };

        const uint32_t maxOpenFilesPerNode = 4;
        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        mThreadParams.dataCollector = new ParallelDomainCollector(
                                                                  gc.getCommunicator().getMPIComm(),
                                                                  gc.getCommunicator().getMPIInfo(),
                                                                  splashMpiSize,
                                                                  maxOpenFilesPerNode);

        mThreadParams.currentStep = restartStep;

        /* set attributes for datacollector files */
        DataCollector::FileCreationAttr attr;
        attr.fileAccType = DataCollector::FAT_READ;
        attr.mpiPosition.set(splashMpiPos);
        attr.mpiSize.set(splashMpiSize);

        /* if restartFilename is relative, prepend with restartDirectory */
        if (!boost::filesystem::path(restartFilename).has_root_path())
        {
            restartFilename = restartDirectory + std::string("/") + restartFilename;
        }

        /* open datacollector */
        try
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 open DataCollector with file: %1%") % restartFilename;
            mThreadParams.dataCollector->open(restartFilename.c_str(), attr);
        }
        catch (const DCException& e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("HDF5 failed to open DataCollector");
        }

        /* load number of slides to initialize MovingWindow */
        uint32_t slides = 0;
        mThreadParams.dataCollector->readAttributeInfo(restartStep, nullptr, "sim_slides").read(&slides, sizeof(slides));

        /* apply slides to set gpus to last/written configuration */
        log<picLog::INPUT_OUTPUT > ("HDF5 setting slide count for moving window to %1%") % slides;
        MovingWindow::getInstance().setSlideCounter(slides, restartStep);

        /* re-distribute the local offsets in y-direction
         * this will work for restarts with moving window still enabled
         * and restarts that disable the moving window
         * \warning enabling the moving window from a checkpoint that
         *          had no moving window will not work
         */
        gc.setStateAfterSlides(slides);

        /* set window for restart, complete global domain */
        mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(restartStep);
        for (uint32_t i = 0; i < simDim; ++i)
        {
            mThreadParams.localWindowToDomainOffset[i] = 0;
        }

        ThreadParams *params = &mThreadParams;

        /* load all fields */
        ForEach<FileCheckpointFields, LoadFields<bmpl::_1> > forEachLoadFields;
        forEachLoadFields(params);

        /* load all particles */
        ForEach<FileCheckpointParticles, LoadSpecies<bmpl::_1> > forEachLoadSpecies;
        forEachLoadSpecies(params, restartChunkSize);

        IdProvider<simDim>::State idProvState;
        ReadNDScalars<uint64_t, uint64_t>()(mThreadParams,
                "picongpu/idProvider/startId", &idProvState.startId,
                "maxNumProc", &idProvState.maxNumProc);
        ReadNDScalars<uint64_t>()(mThreadParams,
                "picongpu/idProvider/nextId", &idProvState.nextId);
        log<picLog::INPUT_OUTPUT > ("Setting next free id on current rank: %1%") % idProvState.nextId;
        IdProvider<simDim>::setState(idProvState);

        /* close datacollector */
        log<picLog::INPUT_OUTPUT > ("HDF5 close DataCollector with file: %1%") % restartFilename;
        mThreadParams.dataCollector->close();

        if (mThreadParams.dataCollector)
            mThreadParams.dataCollector->finalize();

        __delete(mThreadParams.dataCollector);
    }

    void dumpCheckpoint(
        const uint32_t currentStep,
        const std::string& checkpointDirectory,
        const std::string& checkpointFilename
    )
    {
        // checkpointing is only allowed if the plugin is controlled by the class Checkpoint
        assert(!m_help->selfRegister);

        __getTransactionEvent().waitForFinished();
        /* if file name is relative, prepend with common directory */
        if( boost::filesystem::path(checkpointFilename).has_root_path() )
            mThreadParams.h5Filename = checkpointFilename;
        else
            mThreadParams.h5Filename = checkpointDirectory + "/" + checkpointFilename;

        mThreadParams.window = MovingWindow::getInstance().getDomainAsWindow(currentStep);
        mThreadParams.isCheckpoint = true;

        dumpData(currentStep);
    }

private:

    void closeH5File()
    {
        if (mThreadParams.dataCollector != nullptr)
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 close DataCollector");
            mThreadParams.dataCollector->close();
        }
    }

    void openH5File(const std::string h5Filename)
    {
        const uint32_t maxOpenFilesPerNode = 4;
        if (mThreadParams.dataCollector == nullptr)
        {
            GridController<simDim> &gc = Environment<simDim>::get().GridController();
            mThreadParams.dataCollector = new ParallelDomainCollector(
                                                                      gc.getCommunicator().getMPIComm(),
                                                                      gc.getCommunicator().getMPIInfo(),
                                                                      splashMpiSize,
                                                                      maxOpenFilesPerNode);
        }
        // set attributes for datacollector files
        DataCollector::FileCreationAttr attr;
        attr.enableCompression = false;
        attr.fileAccType = DataCollector::FAT_CREATE;
        attr.mpiPosition.set(splashMpiPos);
        attr.mpiSize.set(splashMpiSize);

        // open datacollector
        try
        {
            log<picLog::INPUT_OUTPUT > ("HDF5 open DataCollector with file: %1%") % h5Filename;
            mThreadParams.dataCollector->open(h5Filename.c_str(), attr);
        }
        catch (const DCException& e)
        {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error("HDF5 failed to open DataCollector");
        }
    }

    /** dump data
     *
     * @param currentStep current simulation step
     * @param isCheckpoint checkpoint notification
     */
    void dumpData(uint32_t currentStep)
    {
        const pmacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        mThreadParams.cellDescription = m_cellDescription;
        mThreadParams.currentStep = currentStep;

        for (uint32_t i = 0; i < simDim; ++i)
        {
            mThreadParams.localWindowToDomainOffset[i] = 0;
            if (mThreadParams.window.globalDimensions.offset[i] > localDomain.offset[i])
            {
                mThreadParams.localWindowToDomainOffset[i] =
                    mThreadParams.window.globalDimensions.offset[i] -
                    localDomain.offset[i];
            }
        }

        openH5File(mThreadParams.h5Filename);

        writeHDF5((void*) &mThreadParams);

        closeH5File();
    }

    template< typename T_ParticleFilter>
    struct CallWriteSpecies
    {

        template<typename Space>
        void operator()(
            const std::vector< std::string > & vectorOfDataSourceNames,
            ThreadParams* params,
            const Space domainOffset
        )
        {
            bool const containsDataSource = plugins::misc::containsObject(
                vectorOfDataSourceNames,
                T_ParticleFilter::getName()
            );

            if( containsDataSource )
            {
                WriteSpecies<
                    T_ParticleFilter
                > writeSpecies;
                writeSpecies(params, domainOffset);
            }

        }
    };

    template< typename T_Field >
    struct CallWriteFields
    {

        void operator()(
            const std::vector< std::string > & vectorOfDataSourceNames,
            ThreadParams* params
        )
        {
            bool const containsDataSource = plugins::misc::containsObject(
                vectorOfDataSourceNames,
                T_Field::getName()
            );

            if( containsDataSource )
            {
                WriteFields<
                    T_Field
                > writeFields;
                writeFields(params);
            }

        }
    };

    void writeHDF5(void *p_args)
    {
        ThreadParams *threadParams = (ThreadParams*) (p_args);

        const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
        DataSpace<simDim> domainOffset(
            subGrid.getGlobalDomain().offset +
            subGrid.getLocalDomain().offset
        );

        std::vector< std::string > vectorOfDataSourceNames;
        if( m_help->selfRegister )
        {
            std::string dateSourceNames = m_help->source.get( m_id );

            vectorOfDataSourceNames = plugins::misc::splitString(
                plugins::misc::removeSpaces( dateSourceNames )
            );
        }

        /* write all fields */
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing fields.");
        if (threadParams->isCheckpoint)
        {
            ForEach<FileCheckpointFields, WriteFields<bmpl::_1> > forEachWriteFields;
            forEachWriteFields(threadParams);
        }
        else
        {
            bool dumpFields = plugins::misc::containsObject(
                vectorOfDataSourceNames,
                "fields_all"
            );
            if( dumpFields )
            {
                ForEach<
                    FileOutputFields,
                    WriteFields< bmpl::_1 >
                > forEachWriteFields;
                forEachWriteFields(threadParams);
            }

            ForEach<
                typename Help::AllFieldSources,
                CallWriteFields<
                    bmpl::_1
                >
            >{}(
                vectorOfDataSourceNames,
                threadParams
            );
        }
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing fields.");

        /* write all particle species */
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) writing particle species.");
        if (threadParams->isCheckpoint)
        {
            ForEach<
                FileCheckpointParticles,
                WriteSpecies<
                    plugins::misc::UnfilteredSpecies< bmpl::_1 >
                >
            > writeSpecies;
            writeSpecies(threadParams, domainOffset);
        }
        else
        {
            bool dumpAllParticles = plugins::misc::containsObject(
                vectorOfDataSourceNames,
                "species_all"
            );

            if( dumpAllParticles )
            {
                ForEach<
                    FileOutputParticles,
                    WriteSpecies<
                        plugins::misc::UnfilteredSpecies< bmpl::_1 >
                    >
                > writeSpecies;
                writeSpecies(threadParams, domainOffset);
            }

            ForEach<
                typename Help::AllEligibleSpeciesSources,
                CallWriteSpecies<
                    bmpl::_1
                >
            >{}(
                vectorOfDataSourceNames,
                threadParams,
                domainOffset
            );

        }
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing particle species.");

        auto idProviderState = IdProvider<simDim>::getState();
        log<picLog::INPUT_OUTPUT>("HDF5: Writing IdProvider state (StartId: %1%, NextId: %2%, maxNumProc: %3%)")
                % idProviderState.startId % idProviderState.nextId % idProviderState.maxNumProc;
        WriteNDScalars<uint64_t, uint64_t>()(*threadParams,
                "picongpu/idProvider/startId", idProviderState.startId,
                "maxNumProc", idProviderState.maxNumProc);
        WriteNDScalars<uint64_t>()(*threadParams,
                "picongpu/idProvider/nextId", idProviderState.nextId);

        // write global meta attributes
        WriteMeta writeMetaAttributes;
        writeMetaAttributes(threadParams);
    }

    ThreadParams mThreadParams;

    std::shared_ptr< Help > m_help;
    size_t m_id;

    MappingDesc *m_cellDescription;

    std::string outputDirectory;

    DataSpace<simDim> mpi_pos;
    DataSpace<simDim> mpi_size;

    Dimensions splashMpiPos;
    Dimensions splashMpiSize;
};

} //namespace hdf5
} //namespace picongpu
