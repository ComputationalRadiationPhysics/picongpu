/**
 * Copyright 2014 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
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

#include <string>
#include <sstream>
#include <splash/splash.h>

#include "types.h"
#include "simulation_defines.hpp"
#include "particles/frame_types.hpp"
#include "dataManagement/DataConnector.hpp"
#include "dimensions/DataSpace.hpp"
#include "dimensions/GridLayout.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldB.hpp"
#include "simulationControl/MovingWindow.hpp"

namespace picongpu
{

namespace hdf5
{

using namespace PMacc;
using namespace splash;

/**
 * Helper class for HDF5Writer for loading particle data.
 *
 * @tparam BufferType type of particles buffer
 */
template<class BufferType>
class RestartParticleLoader
{
private:

    template<class TYPE>
    static void loadParticleData(TYPE **dst,
                                 uint32_t restartStep,
                                 ParallelDomainCollector& dataCollector,
                                 Dimensions &dataSize,
                                 CollectionType&,
                                 std::string name,
                                 uint64_t numParticles,
                                 uint64_t particlesLoadOffset)
    {
        /* allocate memory for particles */
        /* workaround, as "*dst = new TYPE[numParticles]" is treated by some compilers as VLA */
        uint8_t *ptr = new uint8_t[numParticles * sizeof(TYPE)];
        *dst = (TYPE*)ptr;
        memset(*dst, 0, sizeof (TYPE) * numParticles);

        // read particles from file
        dataCollector.read(restartStep,
                           Dimensions(numParticles, 1, 1),
                           Dimensions(particlesLoadOffset, 0, 0),
                           name.c_str(),
                           dataSize,
                           *dst
                           );
    }

public:
    
    static void loadParticles(ThreadParams *params,
                              std::string subGroup,
                              BufferType& particles
                              )
    {
        log<picLog::INPUT_OUTPUT > ("Begin loading species '%1%'") % subGroup;

        GridController<simDim> &gc = Environment<simDim>::get().GridController();
        
        const DomainInformation domInfo;
        const DataSpace<simDim> logicalToPhysicalOffset(
            domInfo.localDomain.offset - params->window.globalDimensions.offset);

        // first, load all data arrays from hdf5 file
        CollectionType *ctFloat;
        CollectionType *ctInt;

        ctFloat = new ColTypeFloat();
        ctInt = new ColTypeInt();

        const std::string name_lookup[] = {"x", "y", "z"};

        Dimensions dim_pos(0, 0, 0);
        Dimensions dim_cell(0, 0, 0);
        Dimensions dim_mom(0, 0, 0);
        Dimensions dim_weighting(0, 0, 0);
#if(ENABLE_RADIATION == 1)
        Dimensions dim_mom_mt1(0, 0, 0);

#endif

        typedef float* ptrFloat;
        typedef int* ptrInt;

        ptrFloat relativePositions[simDim];
        ptrInt cellPositions[simDim];
        ptrFloat momentums[simDim];
        ptrFloat weighting = NULL;

        /* load particles info table entry for this process
           particlesInfo is (part-count, scalar pos, x, y, z) */
        typedef uint64_t uint64Quint[5];
        uint64Quint particlesInfo[gc.getGlobalSize()];
        Dimensions particlesInfoSizeRead;

        params->dataCollector->read(params->currentStep, 
                                    (std::string(subGroup) + std::string("/particles_info")).c_str(),
                                    particlesInfoSizeRead,
                                    particlesInfo);

        assert(particlesInfoSizeRead[0] == gc.getGlobalSize());

        /* search my entry (using my scalar position) in particlesInfo */
        uint64_t particleOffset = 0;
        uint64_t particleCount = 0;
        uint64_t myScalarPos = gc.getScalarPosition();

        for (size_t i = 0; i < particlesInfoSizeRead[0]; ++i)
        {
            if (particlesInfo[i][1] == myScalarPos)
            {
                particleCount = particlesInfo[i][0];
                break;
            }

            particleOffset += particlesInfo[i][0];
        }

        log<picLog::INPUT_OUTPUT > ("Loading %1% particles from offset %2%") %
            (long long unsigned) particleCount % (long long unsigned) particleOffset;

#if(ENABLE_RADIATION == 1)
        ptrFloat momentums_mt1[simDim];
#if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
        Dimensions dim_radiationFlag(0, 0, 0);
        CollectionType *ctBool = new ColTypeBool();
        typedef bool* ptrBool;
        ptrBool radiationFlag = NULL;
        loadParticleData<bool> (&radiationFlag, params->currentStep, *(params->dataCollector),
                                dim_radiationFlag, *ctBool, subGroup + std::string("_radiationFlag"),
                                particleCount, particleOffset);
#endif
#endif


        loadParticleData<float> (&weighting, params->currentStep, *(params->dataCollector),
                                 dim_weighting, *ctFloat, subGroup + std::string("/weighting"),
                                 particleCount, particleOffset);

        assert(weighting != NULL);

        for (uint32_t i = 0; i < simDim; ++i)
        {
            relativePositions[i] = NULL;
            cellPositions[i] = NULL;
            momentums[i] = NULL;
#if(ENABLE_RADIATION == 1)
            momentums_mt1[i] = NULL;
#endif

            // read relative positions for particles in cells
            loadParticleData<float> (&(relativePositions[i]), params->currentStep, *(params->dataCollector),
                                     dim_pos, *ctFloat, subGroup + std::string("/position/") + name_lookup[i],
                                     particleCount, particleOffset);

            // read simulation relative cell positions
            loadParticleData<int > (&(cellPositions[i]), params->currentStep, *(params->dataCollector),
                                    dim_cell, *ctInt, subGroup + std::string("/globalCellIdx/") + name_lookup[i],
                                    particleCount, particleOffset);

            // update simulation relative cell positions from file to
            // gpu-relative positions for new configuration
            for (uint32_t elem = 0; elem < dim_cell.getScalarSize(); ++elem)
                cellPositions[i][elem] -= logicalToPhysicalOffset[i];


            // read momentum of particles
            loadParticleData<float> (&(momentums[i]), params->currentStep, *(params->dataCollector),
                                     dim_mom, *ctFloat, subGroup + std::string("/momentum/") + name_lookup[i],
                                     particleCount, particleOffset);

#if(ENABLE_RADIATION == 1)
            // read old momentum of particles
            loadParticleData<float> (&(momentums_mt1[i]), params->currentStep, *(params->dataCollector),
                                     dim_mom_mt1, *ctFloat, subGroup + std::string("/momentumPrev1/") + name_lookup[i],
                                     particleCount, particleOffset);
#endif

            assert(dim_pos[0] == dim_cell[0] && dim_cell[0] == dim_mom[0]);
            assert(dim_pos[0] == dim_pos.getScalarSize() &&
                   dim_cell[0] == dim_cell.getScalarSize() &&
                   dim_mom[0] == dim_mom.getScalarSize());

            assert(relativePositions[i] != NULL);
            assert(cellPositions[i] != NULL);
            assert(momentums[i] != NULL);

#if(ENABLE_RADIATION == 1)
            assert(momentums_mt1[i] != NULL);
#endif
        }

        // now create frames from loaded data
        typename BufferType::ParticlesBoxType particlesBox = particles.getHostParticlesBox();

        typename BufferType::FrameType * frame(NULL);

        DataSpace<simDim> superCellsCount = particles.getParticlesBuffer().getSuperCellsCount();
        DataSpace<simDim> superCellSize = particles.getParticlesBuffer().getSuperCellSize();

        // copy all read data to frames
        DataSpace<simDim> oldSuperCellPos(DataSpace<simDim>::create(-1));
        uint32_t localId = 0;

        for (uint32_t i = 0; i < dim_pos.getScalarSize(); i++)
        {
            // get super cell

            // gpu-global cell position
            DataSpace<simDim> cellPosOnGPU;

            for (uint32_t d = 0; d < simDim; ++d)
                cellPosOnGPU[d] = cellPositions[d][i];

            // gpu-global super cell position
            DataSpace<simDim> superCellPos = (cellPosOnGPU / superCellSize);

            // get gpu-global super cell offset in cells
            //without guarding (need to calculate cell in supercell)
            DataSpace<simDim> superCellOffset = superCellPos * superCellSize;
            // cell position in super cell
            DataSpace<simDim> cellPosInSuperCell = cellPosOnGPU - superCellOffset;


            superCellPos = superCellPos + (int)GUARD_SIZE; //add GUARD supercells

            for (uint32_t d = 0; d < simDim; ++d)
                assert(superCellPos[d] < superCellsCount[d]);


            // grab next empty frame
            if (superCellPos != oldSuperCellPos || localId == PMacc::math::CT::volume<SuperCellSize>::type::value)
            {
                localId = 0;
                frame = &(particlesBox.getEmptyFrame());
                particlesBox.setAsLastFrame(*frame, superCellPos);
                oldSuperCellPos = superCellPos;
            }

            for (uint32_t d = 0; d < simDim; ++d)
                assert(cellPosInSuperCell[d] < SuperCellSize::toRT()[d]);

            PMacc::lcellId_t localCellId(DataSpaceOperations<simDim>::map(superCellSize, cellPosInSuperCell));

            // write to frame
            assert(localId < (uint32_t)PMacc::math::CT::volume<SuperCellSize>::type::value);
            assert((uint32_t) (localCellId) < PMacc::math::CT::volume<SuperCellSize>::type::value);

            PMACC_AUTO(particle, ((*frame)[localId]));

            particle[localCellIdx_] = localCellId;
            particle[multiMask_] = 1;
            particle[weighting_] = weighting[i];
#if(ENABLE_RADIATION == 1) && ((RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0))
            particle[radiationFlag_] = (radiationFlag[i]);
#endif
            for (uint32_t d = 0; d < simDim; ++d)
            {
                particle[position_][d] = relativePositions[d][i];

                particle[momentum_][d] = momentums[d][i];
#if(ENABLE_RADIATION == 1)
                //!\todo: only use Momentum_mt1 if particle type is electrons
                particle[momentumPrev1_][d] = momentums_mt1[d][i];
#endif

            }
            // increase current id/index in frame (0-255)
            localId++;

        }

        particles.syncToDevice();
        particles.fillAllGaps();

        __getTransactionEvent().waitForFinished();

        // cleanup
        for (uint32_t i = 0; i < simDim; ++i)
        {
            delete momentums[i];
            delete cellPositions[i];
            delete relativePositions[i];
        }

        delete weighting;

        delete ctInt;
        delete ctFloat;
#if(ENABLE_RADIATION == 1)
        for (uint32_t i = 0; i < simDim; ++i)
        {
            delete momentums_mt1[i];
        }
#if(RAD_MARK_PARTICLE>1) || (RAD_ACTIVATE_GAMMA_FILTER!=0)
        delete radiationFlag;
        delete ctBool;
#endif
#endif

        log<picLog::INPUT_OUTPUT > ("Finished loading species '%1%'") % subGroup;
    }
};


/**
 * Hepler class for HDF5Writer (forEach operator) to load a particle species from HDF5
 *
 * @tparam ParticleType particle class to load
 */
template< typename ParticleType >
struct LoadParticles
{
public:

    HDINLINE void operator()(ThreadParams* params)
    {
#ifndef __CUDA_ARCH__
        DataConnector &dc = Environment<>::get().DataConnector();
        ThreadParams *tp = params;

        /* load species without copying data to host */
        ParticleType* particles = &(dc.getData<ParticleType >(
                ParticleType::FrameType::getName(), true));

        /* load particle data */
        RestartParticleLoader<ParticleType>::loadParticles(
                tp,
                std::string("particles/") + ParticleType::FrameType::getName(),
                *particles);

        dc.releaseData(ParticleType::FrameType::getName());
#endif
    }

};

} //namespace hdf5
} //namespace picongpu
