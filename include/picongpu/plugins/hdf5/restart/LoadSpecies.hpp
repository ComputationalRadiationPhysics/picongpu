/* Copyright 2013-2019 Rene Widera, Felix Schmitt
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
#include "picongpu/plugins/ISimulationPlugin.hpp"

#include "picongpu/plugins/output/WriteSpeciesCommon.hpp"
#include "picongpu/plugins/hdf5/restart/LoadParticleAttributesFromHDF5.hpp"

#include "picongpu/plugins/common/particlePatches.hpp"
#include "picongpu/plugins/hdf5/openPMD/patchReader.hpp"

#include <pmacc/compileTime/conversion/MakeSeq.hpp>
#include <pmacc/compileTime/conversion/RemoveFromSeq.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/operations/splitIntoListOfFrames.kernel>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/find.hpp>
#include <boost/type_traits.hpp>
#include <boost/type_traits/is_same.hpp>


namespace picongpu
{

namespace hdf5
{
using namespace pmacc;

using namespace splash;

/** Load species from HDF5 checkpoint file
 *
 * @tparam T_Species type of species
 *
 */
template< typename T_Species >
struct LoadSpecies
{
public:

    typedef T_Species ThisSpecies;
    typedef typename ThisSpecies::FrameType FrameType;
    typedef typename FrameType::ParticleDescription ParticleDescription;
    typedef typename FrameType::ValueTypeSeq ParticleAttributeList;


    /* delete multiMask and localCellIdx in hdf5 particle*/
    typedef bmpl::vector2<multiMask, localCellIdx> TypesToDelete;
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

    /** Load species from HDF5 checkpoint file
     *
     * @param params thread params with domainwriter, ...
     * @param restartChunkSize number of particles processed in one kernel call
     */
    HINLINE void operator()(ThreadParams* params, const uint32_t restartChunkSize)
    {
        std::string const speciesName = FrameType::getName();
        log<picLog::INPUT_OUTPUT > ("HDF5: (begin) load species: %1%") % speciesName;
        DataConnector &dc = Environment<>::get().DataConnector();
        GridController<simDim> &gc = Environment<simDim>::get().GridController();

        const std::string speciesSubGroup(
            std::string("particles/") + speciesName + std::string("/")
        );
        const pmacc::Selection<simDim>& localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
        const pmacc::Selection<simDim>& globalDomain = Environment<simDim>::get().SubGrid().getGlobalDomain();

        // load particle without copying particle data to host
        auto speciesTmp = dc.get< ThisSpecies >( FrameType::getName(), true );

        // count total number of particles on the device
        uint64_cu totalNumParticles = 0;
        uint64_t particleOffset = 0;

        // load particle patches offsets to find own patch
        const std::string particlePatchesPath(
            speciesSubGroup + std::string("particlePatches/")
        );

        // avoid deadlock between not finished pmacc tasks and mpi calls in splash/HDF5
        __getTransactionEvent().waitForFinished();

        // read particle patches
        openPMD::PatchReader patchReader;

        picongpu::openPMD::ParticlePatches particlePatches(
            patchReader(
                params->dataCollector,
                gc.getGlobalSize(),
                simDim,
                params->currentStep,
                particlePatchesPath
            )
        );

        /** search my entry (using my cell offset and my local grid size)
         *
         * \note if you want to restart with a changed GPU configuration, either
         * post-process the particle-patches in the file or implement to find
         * all contributing patches and then filter the particles inside those
         * by position
         *
         * \see plugins/hdf5/WriteSpecies.hpp `WriteSpecies::operator()`
         *      as its counterpart
         */
        const DataSpace<simDim> patchOffset =
            globalDomain.offset +
            params->window.globalDimensions.offset +
            params->window.localDimensions.offset;
        const DataSpace<simDim> patchExtent =
            params->window.localDimensions.size;

        for( size_t i = 0; i < gc.getGlobalSize(); ++i )
        {
            bool exactlyMyPatch = true;

            for( uint32_t d = 0; d < simDim; ++d )
            {
                if( particlePatches.getOffsetComp( d )[ i ] != (uint64_t)patchOffset[ d ] )
                    exactlyMyPatch = false;
                if( particlePatches.getExtentComp( d )[ i ] != (uint64_t)patchExtent[ d ] )
                    exactlyMyPatch = false;
            }

            if( exactlyMyPatch )
            {
                totalNumParticles = particlePatches.numParticles[ i ];
                particleOffset = particlePatches.numParticlesOffset[ i ];
                break;
            }
        }

        log<picLog::INPUT_OUTPUT > ("Loading %1% particles from offset %2%") %
            (long long unsigned) totalNumParticles % (long long unsigned) particleOffset;

        Hdf5FrameType hostFrame;
        log<picLog::INPUT_OUTPUT > ("HDF5:  malloc mapped memory: %1%") % speciesName;
        /*malloc mapped memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, MallocMemory<bmpl::_1> > mallocMem;
        mallocMem(forward(hostFrame), totalNumParticles);

        log<picLog::INPUT_OUTPUT > ("HDF5:  get mapped memory device pointer: %1%") % speciesName;
        /*load device pointer of mapped memory*/
        Hdf5FrameType deviceFrame;
        ForEach<typename Hdf5FrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1> > getDevicePtr;
        getDevicePtr(forward(deviceFrame), forward(hostFrame));

        ForEach<typename Hdf5FrameType::ValueTypeSeq, LoadParticleAttributesFromHDF5<bmpl::_1> > loadAttributes;
        loadAttributes(forward(params), forward(hostFrame), speciesSubGroup, particleOffset, totalNumParticles);

        if (totalNumParticles != 0)
        {
            pmacc::particles::operations::splitIntoListOfFrames(
                *speciesTmp,
                deviceFrame,
                totalNumParticles,
                restartChunkSize,
                globalDomain.offset + localDomain.offset,
                totalCellIdx_,
                *(params->cellDescription),
                picLog::INPUT_OUTPUT()
            );

            /*free host memory*/
            ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<bmpl::_1> > freeMem;
            freeMem(forward(hostFrame));
            log<picLog::INPUT_OUTPUT > ("HDF5: ( end ) load species: %1%") % speciesName;
        }
    }
};


} //namspace hdf5

} //namespace picongpu
