/* Copyright 2013-2019 Rene Widera, Felix Schmitt, Axel Huebl
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
#include "picongpu/plugins/hdf5/HDF5Writer.def"
#include "picongpu/traits/SIBaseUnits.hpp"
#include "picongpu/traits/PICToOpenPMD.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/output/WriteSpeciesCommon.hpp"
#include "picongpu/plugins/kernel/CopySpecies.kernel"
#include "picongpu/particles/traits/GetSpeciesFlagName.hpp"
#include "picongpu/plugins/hdf5/writer/ParticleAttribute.hpp"

#include <pmacc/compileTime/conversion/MakeSeq.hpp>
#include <pmacc/compileTime/conversion/RemoveFromSeq.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/traits/HasIdentifier.hpp>
#include <pmacc/assert.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/type_traits.hpp>

#include <string>
#include <type_traits>


namespace picongpu
{

namespace hdf5
{
using namespace pmacc;

// = ColTypeUInt64_5Array
TYPE_ARRAY(UInt64_5, H5T_INTEL_U64, uint64_t, 5);

using namespace splash;

namespace detail
{
    template< typename T_FrameType >
    struct GetChargeOrZero
    {
        static constexpr bool hasChargeRatio = pmacc::traits::HasFlag<
            T_FrameType,
            chargeRatio<>
        >::type::value;

        template< typename T_Defer = float_X >
        typename std::enable_if<
            hasChargeRatio,
            T_Defer
        >::type
        operator()() const
        {
            return frame::getCharge< T_FrameType >();
        }

        template< typename T_Defer = float_X >
        typename std::enable_if<
            !hasChargeRatio,
            T_Defer
        >::type
        operator()() const
        {
            return float_X( 0. );
        }
    };

    template< typename T_FrameType >
    struct GetMassOrZero
    {
        static constexpr bool hasMassRatio = pmacc::traits::HasFlag<
            T_FrameType,
            massRatio<>
        >::type::value;

        template< typename T_Defer = float_X >
        typename std::enable_if<
            hasMassRatio,
            T_Defer
        >::type
        operator()() const
        {
            return frame::getMass< T_FrameType >();
        }

        template< typename T_Defer = float_X >
        typename std::enable_if<
            !hasMassRatio,
            T_Defer
        >::type
        operator()() const
        {
            return float_X( 0. );
        }
    };
}

/** Write copy particle to host memory and dump to HDF5 file
 *
 * @tparam T_SpeciesFilter type and filter of species
 *
 */
template< typename T_SpeciesFilter >
struct WriteSpecies
{
public:

    typedef typename T_SpeciesFilter::Species ThisSpecies;
    typedef typename ThisSpecies::FrameType FrameType;
    typedef typename FrameType::ParticleDescription ParticleDescription;
    typedef typename FrameType::ValueTypeSeq ParticleAttributeList;


    /* delete multiMask and localCellIdx in hdf5 particle*/
    typedef bmpl::vector<multiMask,localCellIdx> TypesToDelete;
    typedef typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type ParticleCleanedAttributeList;

    /* add totalCellIdx for hdf5 particle*/
    typedef typename MakeSeq<
            ParticleCleanedAttributeList,
            totalCellIdx
    >::type ParticleNewAttributeList;

    typedef
    typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type
    NewParticleDescription;

    typedef Frame<OperatorCreateVectorBox, NewParticleDescription> Hdf5FrameType;

    /**
     * @param domainOffset offset to the local domain: globalDomain.offset + localDomain.offset
     */
    template<typename Space>
    HINLINE void operator()(ThreadParams* params,
                            const Space domainOffset)
    {
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) write species: %1%") % T_SpeciesFilter::getName();
        DataConnector &dc = Environment<>::get().DataConnector();
        /* load particle without copy particle data to host */
        auto speciesTmp = dc.get< ThisSpecies >( ThisSpecies::FrameType::getName(), true );

        /* count number of particles for this species on the device */
        uint64_t numParticles = 0;

        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) count particles: %1%") % T_SpeciesFilter::getName();

        // enforce that the filter interface is fulfilled
        particles::filter::IUnary< typename T_SpeciesFilter::Filter > particleFilter{ params->currentStep };
        /* at this point we cast to uint64_t, before we assume that per device
         * less then 1e9 (int range) particles will be counted
         */
        numParticles = uint64_t( pmacc::CountParticles::countOnDevice< CORE + BORDER >(
            *speciesTmp,
            *(params->cellDescription),
            params->localWindowToDomainOffset,
            params->window.localDimensions.size,
            particleFilter
        ));


        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) count particles: %1% = %2%") % T_SpeciesFilter::getName() % numParticles;
        Hdf5FrameType hostFrame;
        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) malloc mapped memory: %1%") % T_SpeciesFilter::getName();
        /*malloc mapped memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, MallocMemory<bmpl::_1> > mallocMem;
        mallocMem(forward(hostFrame), numParticles);
        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) malloc mapped memory: %1%") % T_SpeciesFilter::getName();

        if (numParticles != 0)
        {

            log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) get mapped memory device pointer: %1%") % T_SpeciesFilter::getName();
            /*load device pointer of mapped memory*/
            Hdf5FrameType deviceFrame;
            ForEach<typename Hdf5FrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1> > getDevicePtr;
            getDevicePtr(forward(deviceFrame), forward(hostFrame));
            log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) get mapped memory device pointer: %1%") % T_SpeciesFilter::getName();

            log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) copy particle to host: %1%") % T_SpeciesFilter::getName();
            typedef bmpl::vector< typename GetPositionFilter<simDim>::type > usedFilters;
            typedef typename FilterFactory<usedFilters>::FilterType MyParticleFilter;
            MyParticleFilter filter;
            /* activate filter pipeline if moving window is activated */
            filter.setStatus(MovingWindow::getInstance().isEnabled());
            filter.setWindowPosition(params->localWindowToDomainOffset,
                                     params->window.localDimensions.size);

            /* int: assume < 2e9 particles per device */
            GridBuffer<int, DIM1> counterBuffer(DataSpace<DIM1>(1));
            AreaMapping < CORE + BORDER, MappingDesc > mapper(*(params->cellDescription));

            constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                pmacc::math::CT::volume< SuperCellSize >::type::value
            >::value;

            /* this sanity check costs a little bit of time but hdf5 writing is slower */
            PMACC_KERNEL( CopySpecies< numWorkers >{} )(
                mapper.getGridDim(),
                numWorkers
            )(
                counterBuffer.getDeviceBuffer().getPointer(),
                deviceFrame, speciesTmp->getDeviceParticlesBox(),
                filter,
                domainOffset,
                totalCellIdx_,
                mapper,
                particleFilter
            );
            counterBuffer.deviceToHost();
            log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) copy particle to host: %1%") % T_SpeciesFilter::getName();
            __getTransactionEvent().waitForFinished();
            log<picLog::INPUT_OUTPUT > ("HDF5:  all events are finished: %1%") % T_SpeciesFilter::getName();

            PMACC_ASSERT((uint64_t) counterBuffer.getHostBuffer().getDataBox()[0] == numParticles);
        }

        /* We rather do an allgather at this point then letting libSplash
         * do an allgather during write to find out the global number of
         * particles.
         */
        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) collect particle sizes for %1%") % T_SpeciesFilter::getName();

        ColTypeUInt64 ctUInt64;
        ColTypeDouble ctDouble;
        GridController<simDim>& gc = Environment<simDim>::get().GridController();

        const uint64_t numRanks( gc.getGlobalSize() );
        const uint64_t myRank( gc.getGlobalRank() );

        /* For collective write calls we need the information:
         *   - how many particles will be written globally
         *   - what is my particle offset within this global data set
         *
         * interleaved in array:
         *   numParticles for mpi rank, mpi rank
         *
         * the mpi rank is an arbitrary quantity and might change after a
         * restart, but we only use it to order our patches and offsets
         */
        std::vector<uint64_t> particleCounts( 2 * numRanks, 0u );
        uint64_t myParticlePatch[ 2 ];
        myParticlePatch[ 0 ] = numParticles;
        myParticlePatch[ 1 ] = myRank;

        /* we do the scan over MPI ranks since it does not matter how the
         * global rank or scalar position (which are not idential) are
         * ordered as long as the particle attributes are also written in
         * the same order (which is by global rank) */
        uint64_t numParticlesOffset = 0;
        uint64_t numParticlesGlobal = 0;

        // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
        __getTransactionEvent().waitForFinished();
        MPI_CHECK(MPI_Allgather(
            myParticlePatch, 2, MPI_UINT64_T,
            &(*particleCounts.begin()), 2, MPI_UINT64_T,
            gc.getCommunicator().getMPIComm()
        ));

        for( uint64_t r = 0; r < numRanks; ++r )
        {
            numParticlesGlobal += particleCounts.at(2 * r);
            if( particleCounts.at(2 * r + 1) < myParticlePatch[ 1 ] )
                numParticlesOffset += particleCounts.at(2 * r);
        }
        log<picLog::INPUT_OUTPUT > ("HDF5:  (end) collect particle sizes for %1%") % T_SpeciesFilter::getName();

        /* dump non-constant particle records to hdf5 file */
        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) write particle records for %1%") % T_SpeciesFilter::getName();

        const std::string speciesPath( std::string("particles/") + T_SpeciesFilter::getName() );

        ForEach<typename Hdf5FrameType::ValueTypeSeq, hdf5::ParticleAttribute<bmpl::_1> > writeToHdf5;
        writeToHdf5(
            params,
            forward(hostFrame),
            speciesPath,
            numParticles,
            numParticlesOffset,
            numParticlesGlobal
        );

        /* write constant particle records to hdf5 file
         *   ions with variable charge due to a boundElectrons attribute do not write charge
         */
        using hasBoundElectrons = typename pmacc::traits::HasIdentifier<
            FrameType,
            boundElectrons
        >::type;
        detail::GetChargeOrZero< FrameType > const getChargeOrZero;
        if( ! hasBoundElectrons::value && getChargeOrZero.hasChargeRatio )
        {
            const float_64 charge( getChargeOrZero() );
            std::vector<float_64> chargeUnitDimension( NUnitDimension, 0.0 );
            chargeUnitDimension.at(SIBaseUnits::time) = 1.0;
            chargeUnitDimension.at(SIBaseUnits::electricCurrent) = 1.0;

            writeConstantRecord(
                params,
                speciesPath + std::string("/charge"),
                numParticlesGlobal,
                charge,
                UNIT_CHARGE,
                chargeUnitDimension
            );
        }

        detail::GetMassOrZero< FrameType > const getMassOrZero;
        if( getMassOrZero.hasMassRatio )
        {
            const float_64 mass( getMassOrZero() );
            std::vector<float_64> massUnitDimension( NUnitDimension, 0.0 );
            massUnitDimension.at(SIBaseUnits::mass) = 1.0;

            writeConstantRecord(
                params,
                speciesPath + std::string("/mass"),
                numParticlesGlobal,
                mass,
                UNIT_MASS,
                massUnitDimension
            );
        }

        /* openPMD ED-PIC: write additional attributes */
        const float_64 particleShape( GetShape<ThisSpecies>::type::support - 1 );
        params->dataCollector->writeAttribute( params->currentStep,
                            ctDouble,
                            speciesPath.c_str(),
                            "particleShape",
                            &particleShape );

        traits::GetSpeciesFlagName<ThisSpecies, current<> > currentDepositionName;
        const std::string currentDeposition( currentDepositionName() );
        ColTypeString ctCurrentDeposition( currentDeposition.length() );
        params->dataCollector->writeAttribute( params->currentStep,
                            ctCurrentDeposition,
                            speciesPath.c_str(),
                            "currentDeposition",
                            currentDeposition.c_str() );

        traits::GetSpeciesFlagName<ThisSpecies, particlePusher<> > particlePushName;
        const std::string particlePush( particlePushName() );
        ColTypeString ctParticlePush( particlePush.length() );
        params->dataCollector->writeAttribute( params->currentStep,
                            ctParticlePush,
                            speciesPath.c_str(),
                            "particlePush",
                            particlePush.c_str() );

        traits::GetSpeciesFlagName<ThisSpecies, interpolation<> > particleInterpolationName;
        const std::string particleInterpolation( particleInterpolationName() );
        ColTypeString ctParticleInterpolation( particleInterpolation.length() );
        params->dataCollector->writeAttribute( params->currentStep,
                            ctParticleInterpolation,
                            speciesPath.c_str(),
                            "particleInterpolation",
                            particleInterpolation.c_str() );

        const std::string particleSmoothing("none");
        ColTypeString ctParticleSmoothing(particleSmoothing.length());
        params->dataCollector->writeAttribute( params->currentStep,
                            ctParticleSmoothing,
                            speciesPath.c_str(),
                            "particleSmoothing",
                            particleSmoothing.c_str() );

        log<picLog::INPUT_OUTPUT > ("HDF5:  (end) write particle records for %1%") % T_SpeciesFilter::getName();

        /* write species particle patch meta information */
        log<picLog::INPUT_OUTPUT > ("HDF5:  (begin) writing particlePatches for %1%") % T_SpeciesFilter::getName();

        std::string particlePatchesPath( speciesPath + std::string("/particlePatches") );

        /* offset and size of our particle patches
         *   - numPatches: we write as many patches as MPI ranks
         *   - myPatchOffset: we write in the order of the MPI ranks
         *   - myPatchEntries: every MPI rank writes exactly one patch
         */
        const Dimensions numPatches( numRanks, 1, 1 );
        const Dimensions myPatchOffset( myRank, 0, 0 );
        const Dimensions myPatchEntries( 1, 1, 1 );

        /* numParticles: number of particles in this patch */
        params->dataCollector->write(
            params->currentStep,
            numPatches,
            myPatchOffset,
            ctUInt64, 1,
            myPatchEntries,
            (particlePatchesPath + std::string("/numParticles")).c_str(),
            &numParticles);

        /* numParticlesOffset: number of particles before this patch */
        params->dataCollector->write(
            params->currentStep,
            numPatches,
            myPatchOffset,
            ctUInt64, 1,
            myPatchEntries,
            (particlePatchesPath + std::string("/numParticlesOffset")).c_str(),
            &numParticlesOffset);

        /* offset: absolute position where this particle patch begins including
         *         global domain offsets (slides), etc.
         * extent: size of this particle patch, upper bound is excluded
         */
        const pmacc::Selection<simDim>& globalDomain = Environment<simDim>::get().SubGrid().getGlobalDomain();
        const std::string name_lookup[] = {"x", "y", "z"};
        for (uint32_t d = 0; d < simDim; ++d)
        {
            const uint64_t patchOffset =
                globalDomain.offset[d] +
                params->window.globalDimensions.offset[d] +
                params->window.localDimensions.offset[d];
            const uint64_t patchExtent =
                params->window.localDimensions.size[d];

            params->dataCollector->write(
                params->currentStep,
                numPatches,
                myPatchOffset,
                ctUInt64, 1,
                myPatchEntries,
                (particlePatchesPath + std::string("/offset/") +
                 name_lookup[d]).c_str(),
                &patchOffset);
            params->dataCollector->write(
                params->currentStep,
                numPatches,
                myPatchOffset,
                ctUInt64, 1,
                myPatchEntries,
                (particlePatchesPath + std::string("/extent/") +
                 name_lookup[d]).c_str(),
                &patchExtent);

            /* offsets and extent of the patch are positions (lengths)
             * and need to be scaled like the cell idx of a particle
             */
            OpenPMDUnit<totalCellIdx> openPMDUnitCellIdx;
            std::vector<float_64> unitCellIdx = openPMDUnitCellIdx();

            params->dataCollector->writeAttribute(
                params->currentStep,
                ctDouble,
                (particlePatchesPath + std::string("/offset/") +
                 name_lookup[d]).c_str(),
                "unitSI",
                &(unitCellIdx.at(d)));
            params->dataCollector->writeAttribute(
                params->currentStep,
                ctDouble,
                (particlePatchesPath + std::string("/extent/") +
                 name_lookup[d]).c_str(),
                "unitSI",
                &(unitCellIdx.at(d)));
        }

        OpenPMDUnitDimension<totalCellIdx> openPMDUnitDimension;
        std::vector<float_64> unitDimensionCellIdx = openPMDUnitDimension();

        params->dataCollector->writeAttribute(
            params->currentStep,
            ctDouble,
            (particlePatchesPath + std::string("/offset")).c_str(),
            "unitDimension",
            1u, Dimensions(7,0,0),
            &(*unitDimensionCellIdx.begin()));
        params->dataCollector->writeAttribute(
            params->currentStep,
            ctDouble,
            (particlePatchesPath + std::string("/extent")).c_str(),
            "unitDimension",
            1u, Dimensions(7,0,0),
            &(*unitDimensionCellIdx.begin()));


        log<picLog::INPUT_OUTPUT > ("HDF5:  ( end ) writing particlePatches for %1%") % T_SpeciesFilter::getName();

        /*free host memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<bmpl::_1> > freeMem;
        freeMem(forward(hostFrame));
        log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) writing species: %1%") % T_SpeciesFilter::getName();
    }

private:

    /** Writes a constant particle record (weighted for a real particle)
     *
     * @param params thread parameters
     * @param recordPath path to the record
     * @param numParticlesGlobal global number of particles in the species
     * @param value of the record
     * @param unitSI conversion factor to SI
     * @param unitDimension power in terms of SI base units for this record
     */
    static void writeConstantRecord(
        ThreadParams* params,
        const std::string recordPath,
        const uint64_t numParticlesGlobal,
        const float_64 value,
        const float_64 unitSI,
        const std::vector<float_64>& unitDimension
    )
    {
        typedef typename PICToSplash<float_X>::type SplashFloatXType;

        ColTypeUInt32 ctUInt32;
        ColTypeUInt64 ctUInt64;
        ColTypeDouble ctDouble;
        SplashFloatXType splashFloatXType;

        /* openPMD base standard
         *   write constant record
         */
        params->dataCollector->writeAttribute(
            params->currentStep,
            ctDouble, recordPath.c_str(),
            "value", &value);

        params->dataCollector->writeAttribute(
            params->currentStep,
            ctUInt64, recordPath.c_str(),
            "shape",
            1u, Dimensions(1,0,0),
            &numParticlesGlobal);

        params->dataCollector->writeAttribute(
            params->currentStep,
            ctDouble, recordPath.c_str(),
            "unitSI", &unitSI);

        params->dataCollector->writeAttribute(
            params->currentStep,
            ctDouble, recordPath.c_str(),
            "unitDimension",
            1u, Dimensions(7,0,0),
            &(*unitDimension.begin()));

        /** \todo check if always correct at this point, depends on attribute
         *        and MW-solver/pusher implementation */
        const float_X timeOffset( 0.0 );      // same type as "time" in basePath
        params->dataCollector->writeAttribute(
            params->currentStep,
            splashFloatXType, recordPath.c_str(),
            "timeOffset", &timeOffset);

        /* ED-PIC extension:
         *   - this is a record describing a *real* particle (0: false)
         *   - it needs to be scaled linearly (w^1.0) to get the *macro*
         *     particle record
         */
        const uint32_t macroWeighted( 0 );
        params->dataCollector->writeAttribute(
            params->currentStep,
            ctUInt32, recordPath.c_str(),
            "macroWeighted",
            &macroWeighted);

        const float_64 weightingPower( 1.0 );
        params->dataCollector->writeAttribute(
            params->currentStep,
            ctDouble, recordPath.c_str(),
            "weightingPower",
            &weightingPower);
    }
};


} //namspace hdf5

} //namespace picongpu
