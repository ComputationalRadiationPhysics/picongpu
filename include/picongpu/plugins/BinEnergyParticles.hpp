/* Copyright 2013-2017 Axel Huebl, Felix Schmitt, Heiko Burau,
 *                     Rene Widera, Richard Pausch
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

#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/algorithms/Gamma.hpp"
#include "picongpu/algorithms/KinEnergy.hpp"

#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/mappings/threads/ForEachIdx.hpp>
#include <pmacc/mappings/threads/IdxConfig.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/nvidia/atomic.hpp>

#include "common/txtFileHandling.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>


namespace picongpu
{
using namespace pmacc;

namespace po = boost::program_options;

/** calculate a energy histogram of a species
 *
 * if a detector is defined than only particle traversing the detector are counted
 * else all particles are used
 *
 * @tparam T_numWorkers number of workers
 */
template< uint32_t T_numWorkers >
struct KernelBinEnergyParticles
{
    /* sum up the energy of all particles
     *
     * the kinetic energy of all active particles will be calculated
     *
     * @tparam T_ParBox pmacc::ParticlesBox, particle box type
     * @tparam T_BinBox pmacc::DataBox, box type for the histogram in global memory
     * @tparam T_Mapping type of the mapper to map a cuda block to a supercell index
     * @tparam T_Acc alpaka accelerator type
     *
     * @param acc alpaka accelerator
     * @param pb box with access to the particles of the current used species
     * @param gBins box with memory for resulting histogram
     * @param numBins number of bins in the histogram (must be fit into the shared memory)
     * @param minEnergy particle energy for the first bin
     * @param maxEnergy particle energy for the last bin
     * @param maximumSlopeToDetectorX tangent of maximum opening angle to the detector in X direction
     * @param maximumSlopeToDetectorZ tangent of maximum opening angle to the detector in Y direction
     * @param mapper functor to map a cuda block to a supercells index
     */
    template<
        typename T_ParBox,
        typename T_BinBox,
        typename T_Mapping,
        typename T_Acc
    >
    DINLINE void operator()(
        T_Acc const & acc,
        T_ParBox pb,
        T_BinBox gBins,
        int const numBins,
        float_X const minEnergy,
        float_X const maxEnergy,
        float_X const maximumSlopeToDetectorX,
        float_X const maximumSlopeToDetectorZ,
        T_Mapping const mapper
    ) const
    {
        using namespace pmacc::mappings::threads;
        using SuperCellSize = typename MappingDesc::SuperCellSize;
        using FramePtr = typename T_ParBox::FramePtr;
        constexpr uint32_t maxParticlesPerFrame = pmacc::math::CT::volume< SuperCellSize >::type::value;
        constexpr uint32_t numWorkers = T_numWorkers;

        PMACC_SMEM(
            acc,
            frame,
            FramePtr
        );

        PMACC_SMEM(
            acc,
            particlesInSuperCell,
            lcellId_t
        );

        bool const enableDetector = maximumSlopeToDetectorX != float_X( 0.0 ) && maximumSlopeToDetectorZ != float_X( 0.0 );

        /* shBins index can go from 0 to (numBins+2)-1
         * 0 is for <minEnergy
         * (numBins+2)-1 is for >maxEnergy
         */
        sharedMemExtern(shBin,float_X); /* size must be numBins+2 because we have <min and >max */


        int const realNumBins = numBins + 2;

        uint32_t const workerIdx = threadIdx.x;

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
                DataSpace< simDim > const superCellIdx(
                    mapper.getSuperCellIndex( DataSpace< simDim >( blockIdx ) )
                );
                frame = pb.getLastFrame( superCellIdx );
                particlesInSuperCell = pb.getSuperCell( superCellIdx ).getSizeLastFrame( );
            }
        );

        ForEachIdx<
            IdxConfig<
                numWorkers,
                numWorkers
            >
        >{ workerIdx }(
            [&](
                uint32_t const linearIdx,
                uint32_t const
            )
            {
                /* set all bins to 0 */
                for( int i = linearIdx; i < realNumBins; i += numWorkers )
                    shBin[ i ] = float_X( 0. );
            }
        );

        __syncthreads();

        if( !frame.isValid( ) )
          return; /* end kernel if we have no frames */

        while( frame.isValid() )
        {
            // move over all particles in a frame
            ForEachIdx<
                IdxConfig<
                    maxParticlesPerFrame,
                    numWorkers
                >
            >{ workerIdx }(
                [&](
                    uint32_t const linearIdx,
                    uint32_t const
                )
                {
                    if( linearIdx < particlesInSuperCell )
                    {
                        auto const particle = frame[ linearIdx ];
                        /* kinetic Energy for Particles: E^2 = p^2*c^2 + m^2*c^4
                         *                                   = c^2 * [p^2 + m^2*c^2]
                         */
                        float3_X const mom = particle[ momentum_ ];

                        bool calcParticle = true;

                        if( enableDetector && mom.y() > 0.0 )
                        {
                            float_X const slopeMomX = abs( mom.x( ) / mom.y( ) );
                            float_X const slopeMomZ = abs( mom.z( ) / mom.y( ) );
                            if( slopeMomX >= maximumSlopeToDetectorX || slopeMomZ >= maximumSlopeToDetectorZ )
                                calcParticle = false;
                        }

                        if( calcParticle )
                        {
                            float_X const weighting = particle[ weighting_ ];
                            float_X const mass = attribute::getMass(
                                weighting,
                                particle
                            );

                            // calculate kinetic energy of the macro particle
                            float_X localEnergy = KinEnergy< >( )(
                                mom,
                                mass
                            );

                            localEnergy /= weighting;

                            /* +1 move value from 1 to numBins+1 */
                            int binNumber = math::floor(
                                ( localEnergy - minEnergy ) /
                                ( maxEnergy - minEnergy ) * static_cast< float_X >( numBins )
                            )  + 1;

                            int const maxBin = numBins + 1;

                            /* all entries larger than maxEnergy go into bin maxBin */
                            binNumber = binNumber < maxBin ? binNumber : maxBin;

                            /* all entries smaller than minEnergy go into bin zero */
                            binNumber = binNumber > 0 ? binNumber : 0;

                            /*!\todo: we can't use 64bit type on this place (NVIDIA BUG?)
                             * COMPILER ERROR: ptxas /tmp/tmpxft_00005da6_00000000-2_main.ptx, line 4246; error   : Global state space expected for instruction 'atom'
                             * I think this is a problem with extern shared mem and atmic (only on TESLA)
                             * NEXT BUG: don't do uint32_t w=__float2uint_rn(weighting); and use w for atomic, this create wrong results
                             *
                             * uses a normed float weighting to avoid a overflow of the floating point result
                             * for the reduced weighting if the particle weighting is very large
                             */
                            float_X const normedWeighting = weighting /
                                float_X( particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE );
                            atomicAdd(
                                &( shBin[ binNumber ] ),
                                normedWeighting,
                                ::alpaka::hierarchy::Threads{}
                            );
                        }
                    }
                }
            );

            __syncthreads();

            ForEachIdx< MasterOnly >{ workerIdx }(
                [&](
                    uint32_t const,
                    uint32_t const
                )
                {
                    frame = pb.getPreviousFrame( frame );
                    particlesInSuperCell = maxParticlesPerFrame;
                }
            );
            __syncthreads();
        }

        ForEachIdx<
            IdxConfig<
                numWorkers,
                numWorkers
            >
        >{ workerIdx }(
            [&](
                uint32_t const linearIdx,
                uint32_t const
            )
            {
                for( int i = linearIdx; i < realNumBins; i += numWorkers )
                    atomicAdd(
                        &( gBins[ i ] ),
                        float_64( shBin[ i ] ),
                        ::alpaka::hierarchy::Blocks{}
                    );
            }
        );
    }
};

template<class ParticlesType>
class BinEnergyParticles : public ISimulationPlugin
{
private:

    typedef MappingDesc::SuperCellSize SuperCellSize;

    GridBuffer<float_64, DIM1> *gBins;
    MappingDesc *cellDescription;

    std::string pluginName;
    std::string pluginPrefix;
    std::string filename;

    float_64 * binReduced;

    uint32_t notifyPeriod;
    int numBins;
    int realNumBins;
    /* variables for energy limits of the histogram in keV */
    float_X minEnergy_keV;
    float_X maxEnergy_keV;

    float_X distanceToDetector;
    float_X slitDetectorX;
    float_X slitDetectorZ;
    bool enableDetector;

    std::ofstream outFile;

    /* only rank 0 create a file */
    bool writeToFile;

    mpi::MPIReduce reduce;

public:

    BinEnergyParticles() :
    pluginName("BinEnergyParticles: calculate a energy histogram of a species"),
    pluginPrefix(ParticlesType::FrameType::getName() + std::string("_energyHistogram")),
    filename(pluginPrefix + ".dat"),
    gBins(nullptr),
    cellDescription(nullptr),
    notifyPeriod(0),
    writeToFile(false),
    enableDetector(false)
    {
        Environment<>::get().PluginConnector().registerPlugin(this);
    }

    virtual ~BinEnergyParticles()
    {

    }

    void notify(uint32_t currentStep)
    {
        calBinEnergyParticles < CORE + BORDER > (currentStep);
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((pluginPrefix + ".period").c_str(), po::value<uint32_t > (&notifyPeriod)->default_value(0), "enable plugin [for each n-th step]")
            ((pluginPrefix + ".binCount").c_str(), po::value<int > (&numBins)->default_value(1024), "number of bins for the energy range")
            ((pluginPrefix + ".minEnergy").c_str(), po::value<float_X > (&minEnergy_keV)->default_value(0.0), "minEnergy[in keV]")
            ((pluginPrefix + ".maxEnergy").c_str(), po::value<float_X > (&maxEnergy_keV), "maxEnergy[in keV]")
            ((pluginPrefix + ".distanceToDetector").c_str(), po::value<float_X > (&distanceToDetector)->default_value(0.0), "distance between gas and detector, assumptions: simulated area in y direction << distance to detector AND simulated area in X,Z << slit [in meters]  (if not set, all particles are counted)")
            ((pluginPrefix + ".slitDetectorX").c_str(), po::value<float_X > (&slitDetectorX)->default_value(0.0), "size of the detector slit in X [in meters] (if not set, all particles are counted)")
            ((pluginPrefix + ".slitDetectorZ").c_str(), po::value<float_X > (&slitDetectorZ)->default_value(0.0), "size of the detector slit in Z [in meters] (if not set, all particles are counted)");
    }

    std::string pluginGetName() const
    {
        return pluginName;
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        this->cellDescription = cellDescription;
    }

private:

    /* Open a New Output File
     *
     * Must only be called by the rank with writeToFile == true
     */
    void openNewFile()
    {
        outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
        if (!outFile)
        {
            std::cerr << "[Plugin] [" << pluginPrefix
                      << "] Can't open file '" << filename
                      << "', output disabled" << std::endl;
            writeToFile = false;
        }
        else
        {
            /* create header of the file */
            outFile << "#step <" << minEnergy_keV << " ";
            float_X binEnergy = (maxEnergy_keV - minEnergy_keV) / (float_32) numBins;
            for (int i = 1; i < realNumBins - 1; ++i)
                outFile << minEnergy_keV + ((float_32) i * binEnergy) << " ";

            outFile << ">" << maxEnergy_keV << " count" << std::endl;
        }
    }

    void pluginLoad()
    {
        if (notifyPeriod > 0)
        {
            if( numBins <= 0 )
            {
                std::cerr << "[Plugin] [" << pluginPrefix
                          << "] disabled since " << pluginPrefix
                          << ".binCount must be > 0 (input "
                          << numBins << " bins)"
                          << std::endl;

                /* do not register the plugin and return */
                return;
            }

            if (distanceToDetector != float_X(0.0) && slitDetectorX != float_X(0.0) && slitDetectorZ != float_X(0.0))
                enableDetector = true;

            realNumBins = numBins + 2;

            /* create an array of float_64 on gpu und host */
            gBins = new GridBuffer<float_64, DIM1 > (DataSpace<DIM1 > (realNumBins));
            binReduced = new float_64[realNumBins];
            for (int i = 0; i < realNumBins; ++i)
            {
                binReduced[i] = 0.0;
            }

            writeToFile = reduce.hasResult(mpi::reduceMethods::Reduce());
            if( writeToFile )
                openNewFile();

            Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
        }
    }

    void pluginUnload()
    {
        if (notifyPeriod > 0)
        {
            if (writeToFile)
            {
                outFile.flush();
                outFile << std::endl; /* now all data are written to file */
                if (outFile.fail())
                    std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                outFile.close();
            }

            __delete(gBins);
            __deleteArray(binReduced);
        }
    }

    void restart(uint32_t restartStep, const std::string restartDirectory)
    {
        if( !writeToFile )
            return;

        writeToFile = restoreTxtFile( outFile,
                                      filename,
                                      restartStep,
                                      restartDirectory );
    }

    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
        if( !writeToFile )
            return;

        checkpointTxtFile( outFile,
                           filename,
                           currentStep,
                           checkpointDirectory );
    }

    template< uint32_t AREA>
    void calBinEnergyParticles(uint32_t currentStep)
    {
        gBins->getDeviceBuffer().setValue(0);

        DataConnector &dc = Environment<>::get().DataConnector();
        auto particles = dc.get< ParticlesType >( ParticlesType::FrameType::getName(), true );

        /** Assumption: distanceToDetector >> simulated Area in y-Direction
         *          AND     simulated area in X,Z << slit  */
        float_64 maximumSlopeToDetectorX = 0.0; /*0.0 is disabled detector*/
        float_64 maximumSlopeToDetectorZ = 0.0; /*0.0 is disabled detector*/
        if (enableDetector)
        {
            maximumSlopeToDetectorX = (slitDetectorX / 2.0) / (distanceToDetector);
            maximumSlopeToDetectorZ = (slitDetectorZ / 2.0) / (distanceToDetector);
            /* maximumSlopeToDetector = (radiusDetector * radiusDetector) / (distanceToDetector * distanceToDetector); */
        }

        /* convert energy values from keV to PIConGPU units */
        float_X const minEnergy = minEnergy_keV * UNITCONV_keV_to_Joule / UNIT_ENERGY;
        float_X const maxEnergy = maxEnergy_keV * UNITCONV_keV_to_Joule / UNIT_ENERGY;

        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
             pmacc::math::CT::volume< SuperCellSize >::type::value
         >::value;

        AreaMapping<
            AREA,
            MappingDesc
        > mapper( *cellDescription );

        PMACC_KERNEL( KernelBinEnergyParticles< numWorkers >{ } )(
            mapper.getGridDim(),
            numWorkers,
            realNumBins * sizeof( float_X )
        )(
            particles->getDeviceParticlesBox( ),
            gBins->getDeviceBuffer( ).getDataBox( ),
            numBins,
            minEnergy,
            maxEnergy,
            maximumSlopeToDetectorX,
            maximumSlopeToDetectorZ,
            mapper
        );

        dc.releaseData( ParticlesType::FrameType::getName() );
        gBins->deviceToHost();

        reduce(nvidia::functors::Add(),
               binReduced,
               gBins->getHostBuffer().getBasePointer(),
               realNumBins, mpi::reduceMethods::Reduce());


        if (writeToFile)
        {
            typedef std::numeric_limits< float_64 > dbl;

            outFile.precision(dbl::digits10);

            /* write data to file */
            float_64 count_particles = 0.0;
            outFile << currentStep << " "
                    << std::scientific; /*  for floating points, ignored for ints */

            for (int i = 0; i < realNumBins; ++i)
            {
                count_particles += float_64( binReduced[i]);
                outFile << std::scientific << (binReduced[i]) * float_64(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE) << " ";
            }
            outFile << std::scientific << count_particles * float_64(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE)
                << std::endl;
            /* endl: Flush any step to the file.
             * Thus, we will have data if the program should crash. */
        }
    }

};

}



