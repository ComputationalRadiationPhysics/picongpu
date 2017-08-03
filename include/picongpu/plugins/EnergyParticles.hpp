/* Copyright 2013-2017 Axel Huebl, Felix Schmitt, Heiko Burau,
 *                     Rene Widera, Richard Pausch, Benjamin Worpitz
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
#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/common/txtFileHandling.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/nvidia/atomic.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>
#include <pmacc/memory/CtxArray.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <string>
#include <iostream>
#include <fstream>


namespace picongpu
{

    /** accumulate the kinetic and total energy
     *
     * All energies are summed over all particles of a species.
     *
     * @tparam T_numWorkers number of workers
     */
    template< uint32_t T_numWorkers >
    struct KernelEnergyParticles
    {

        /** accumulate particle energies
         *
         * @tparam T_ParBox pmacc::ParticlesBox, particle box type
         * @tparam T_DBox pmacc::DataBox, type of the memory box for the reduced energies
         * @tparam T_Mapping mapper functor type
         *
         * @param pb particle memory
         * @param gEnergy storage for the reduced energies
         *                (two elements 0 == kinetic; 1 == total energy)
         * @param mapper functor to map a block to a supercell
         */
        template<
            typename T_ParBox,
            typename T_DBox,
            typename T_Mapping,
            typename T_Acc
        >
        DINLINE void operator( )(
            T_Acc const & acc,
            T_ParBox pb,
            T_DBox gEnergy,
            T_Mapping mapper
        ) const
        {
            using namespace mappings::threads;

            constexpr uint32_t numWorkers = T_numWorkers;
            constexpr uint32_t numParticlesPerFrame = pmacc::math::CT::volume<
                typename T_ParBox::FrameType::SuperCellSize
            >::type::value;

            uint32_t const workerIdx = threadIdx.x;

            using FramePtr = typename T_ParBox::FramePtr;

            // shared kinetic energy
            PMACC_SMEM(
                acc,
                shEnergyKin,
                float_X
            );
            // shared total energy
            PMACC_SMEM(
                acc,
                shEnergy,
                float_X
            );

            using ParticleDomCfg = IdxConfig<
                numParticlesPerFrame,
                numWorkers
            >;

            // sum kinetic energy for all particles touched by the virtual thread
            float_X localEnergyKin( 0.0 );
            float_X localEnergy( 0.0 );

            using MasterOnly = IdxConfig<
                1,
                numWorkers
            >;


            ForEachIdx< MasterOnly >{ workerIdx }(
                [&](
                    uint32_t const,
                    uint32_t const
                )
                {
                    // set shared kinetic energy to zero
                    shEnergyKin = float_X( 0.0 );
                    // set shared total energy to zero
                    shEnergy = float_X( 0.0 );
                }
            );

            __syncthreads( );

            DataSpace< simDim > const superCellIdx( mapper.getSuperCellIndex(
                DataSpace< simDim >( blockIdx )
            ));

            // each virtual thread is working on an own frame
            FramePtr frame = pb.getLastFrame( superCellIdx );

            // end kernel if we have no frames within the supercell
            if( !frame.isValid( ) )
                return;

            memory::CtxArray<
                bool,
                ParticleDomCfg
            >
            isParticleCtx(
                workerIdx,
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    return frame[ linearIdx ][ multiMask_ ];
                }
            );

            while( frame.isValid( ) )
            {
                // loop over all particles in the frame
                ForEachIdx< ParticleDomCfg > forEachParticle( workerIdx );

                forEachParticle(
                    [&](
                        uint32_t const linearIdx,
                        uint32_t const idx
                    )
                    {
                        if( isParticleCtx[ idx ] )
                        {
                            /* get one particle */
                            auto particle = frame[ linearIdx ];
                            float3_X const mom = particle[ momentum_ ];
                            // compute square of absolute momentum of the particle
                            float_X const mom2 = math::abs2( mom );
                            float_X const weighting = particle[ weighting_ ];
                            float_X const mass = attribute::getMass(
                                weighting,
                                particle
                            );
                            float_X const c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

                            // calculate kinetic energy of the macro particle
                            localEnergyKin += KinEnergy<>( )(
                                mom,
                                mass
                            );

                            /* total energy for particles:
                             *    E^2 = p^2*c^2 + m^2*c^4
                             *        = c^2 * [p^2 + m^2*c^2]
                             */
                            localEnergy += algorithms::math::sqrt(
                                mom2 +
                                mass * mass * c2
                            ) * SPEED_OF_LIGHT;

                        }
                    }
                );

                // set frame to next particle frame
                frame = pb.getPreviousFrame(frame);
                forEachParticle(
                    [&](
                        uint32_t const,
                        uint32_t const idx
                    )
                    {
                        /* The frame list is traverse from the last to the first frame.
                         * Only the last frame can contain gaps therefore all following
                         * frames are filled with fully particles
                         */
                        isParticleCtx[ idx ] = true;
                    }
                );
            }

            // each virtual thread adds the energies to the shared memory
            atomicAdd(
                &shEnergyKin,
                localEnergyKin,
                ::alpaka::hierarchy::Threads{}
            );
            atomicAdd(
                &shEnergy,
                localEnergy,
                ::alpaka::hierarchy::Threads{}
            );

            // wait that all virtual threads updated the shared memory energies
            __syncthreads( );

            // add energies on global level using global memory
            ForEachIdx< MasterOnly >{ workerIdx }(
                [&](
                    uint32_t const,
                    uint32_t const
                )
                {
                    // add kinetic energy
                    atomicAdd(
                        &( gEnergy[ 0 ] ),
                        static_cast< float_64 >( shEnergyKin ),
                        ::alpaka::hierarchy::Blocks{}
                    );
                    // add total energy
                    atomicAdd(
                        &( gEnergy[ 1 ] ),
                        static_cast< float_64 >( shEnergy ),
                        ::alpaka::hierarchy::Blocks{}
                    );
                }
            );
        }
    };

    template< typename ParticlesType >
    class EnergyParticles : public ISimulationPlugin
    {
    private:
        //! energy values (global on GPU)
        GridBuffer<
            float_64,
            DIM1
        > * gEnergy;
        MappingDesc * cellDescription;

        //! periodicity of computing the particle energy
        uint32_t notifyPeriod;

        //! name (used for output file too)
        std::string pluginName;

        //! prefix used for command line arguments
        std::string pluginPrefix;

        //! output file name
        std::string filename;

        //! file output stream
        std::ofstream outFile;

        /** only one MPI rank creates a file
         *
         * true if this MPI rank creates the file, else false
         */
        bool writeToFile;

        //! MPI reduce to add all energies over several GPUs
        mpi::MPIReduce reduce;

    public:

        EnergyParticles( ) :
            pluginName( "EnergyParticles: calculate the energy of a species" ),
            pluginPrefix( ParticlesType::FrameType::getName( ) + std::string( "_energy" ) ),
            filename( pluginPrefix + ".dat" ),
            gEnergy( nullptr ),
            cellDescription( nullptr ),
            notifyPeriod( 0 ),
            writeToFile( false )
        {
            // register this plugin
            Environment< >::get( ).PluginConnector( ).registerPlugin( this );
        }

        virtual ~EnergyParticles( )
        {

        }

        /** this code is executed if the current time step is supposed to compute
         * the energy
         */
        void notify( uint32_t currentStep )
        {
            // call the method that calls the plugin kernel
            calculateEnergyParticles < CORE + BORDER > ( currentStep );
        }

        ///! method used by plugin controller to get --help description
        void pluginRegisterHelp( boost::program_options::options_description& desc )
        {
            desc.add_options( )(
                ( pluginPrefix + ".period").c_str( ),
                boost::program_options::value< uint32_t >( &notifyPeriod ),
                "compute kinetic and total energy [for each n-th step] enable plugin by setting a non-zero value"
            );
        }

        //! method giving the plugin name (used by plugin control)
        std::string pluginGetName( ) const
        {
            return pluginName;
        }

        //! set cell description in this plugin
        void setMappingDescription( MappingDesc *cellDescription )
        {
            this->cellDescription = cellDescription;
        }

    private:

        //! method to initialize plugin output and variables
        void pluginLoad( )
        {
            // only if plugin is called at least once
            if( notifyPeriod > 0 )
            {
                // decide which MPI-rank writes output
                writeToFile = reduce.hasResult( mpi::reduceMethods::Reduce( ) );

                // create two ints on gpu and host
                gEnergy = new GridBuffer<
                    float_64,
                    DIM1
                >( DataSpace< DIM1 >( 2 ) );

                // only MPI rank that writes to file
                if( writeToFile )
                {
                    // open output file
                    outFile.open(
                        filename.c_str( ),
                        std::ofstream::out | std::ostream::trunc
                    );

                    // error handling
                    if( !outFile )
                    {
                        std::cerr <<
                            "Can't open file [" <<
                            filename <<
                            "] for output, diasble plugin output. " <<
                            std::endl;
                        writeToFile = false;
                    }

                    // create header of the file
                    outFile << "#step Ekin_Joule E_Joule" << " \n";
                }

                // set how often the plugin should be executed while PIConGPU is running
                Environment<>::get( ).PluginConnector( ).setNotificationPeriod(
                    this,
                    notifyPeriod
                );
            }
        }

        //! method to quit plugin
        void pluginUnload( )
        {
            // only if plugin is called at least once
            if( notifyPeriod > 0 )
            {
                if( writeToFile )
                {
                    outFile.flush( );
                    // flush cached data to file
                    outFile << std::endl;

                    if( outFile.fail( ) )
                        std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                    outFile.close( );
                }
                // free global memory on GPU
                __delete( gEnergy );
            }
        }

        void restart(
            uint32_t restartStep,
            std::string const restartDirectory
        )
        {
            if( !writeToFile )
                return;

            writeToFile = restoreTxtFile(
                outFile,
                filename,
                restartStep,
                restartDirectory
            );
        }

        void checkpoint(
            uint32_t currentStep,
            std::string const checkpointDirectory
        )
        {
            if( !writeToFile )
                return;

            checkpointTxtFile(
                outFile,
                filename,
                currentStep,
                checkpointDirectory
            );
        }

        //! method to call analysis and plugin-kernel calls
        template< uint32_t AREA >
        void calculateEnergyParticles( uint32_t currentStep )
        {
            DataConnector &dc = Environment<>::get( ).DataConnector( );

            // use data connector to get particle data
            auto particles = dc.get< ParticlesType >(
                ParticlesType::FrameType::getName( ),
                true
            );

            // initialize global energies with zero
            gEnergy->getDeviceBuffer( ).setValue( 0.0 );

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            AreaMapping<
                AREA,
                MappingDesc
            > mapper( *cellDescription );

            PMACC_KERNEL( KernelEnergyParticles< numWorkers >{ } )(
                mapper.getGridDim( ),
                numWorkers
            )(
                particles->getDeviceParticlesBox( ),
                gEnergy->getDeviceBuffer( ).getDataBox( ),
                mapper
            );

            dc.releaseData( ParticlesType::FrameType::getName( ) );

            // get energy from GPU
            gEnergy->deviceToHost( );

            // create storage for the global reduced result
            float_64 reducedEnergy[2];

            // add energies from all GPUs using MPI
            reduce(
                nvidia::functors::Add( ),
                reducedEnergy,
                gEnergy->getHostBuffer( ).getBasePointer( ),
                2,
                mpi::reduceMethods::Reduce( )
            );

            /* print timestep, kinetic energy and total energy to file: */
            if( writeToFile )
            {
                using dbl = std::numeric_limits< float_64 >;

                outFile.precision( dbl::digits10 );
                outFile << currentStep << " "
                        << std::scientific
                        << reducedEnergy[ 0 ] * UNIT_ENERGY << " "
                        << reducedEnergy[ 1 ] * UNIT_ENERGY << std::endl;
            }
        }

    };

} // namespace picongpu
