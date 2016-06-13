/**
 * Copyright 2013-2016 Rene Widera, Felix Schmitt, Alexander Grund
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

#include "pmacc_types.hpp"
#include "dimensions/DataSpaceOperations.hpp"
#include "math/vector/compile-time/Vector.hpp"

namespace PMacc
{
namespace particles
{
namespace operations
{

/** Copy Particles to a Single Frame
 *
 * - copy particle data that was stored in a linked list of frames for each
 *   super-cell on the GPU to a single frame on the CPU RAM
 * - the deep on-GPU hierarchy must be copied to the CPU beforehand
 * - remove species attributes `multiMask` and `localCellIdx`
 * - add new attribute `globalCellIdx` (particle offset to begin of global
 *   moving window)
 */
template<unsigned T_dim>
struct ConcatListOfFrames
{
    DataSpace<T_dim> m_gridDim;

    ConcatListOfFrames(const DataSpace<T_dim>& gridDim) :
    m_gridDim(gridDim)
    {

    }

    /** concatenate list of frames to single frame
     *
     * @param counter[in,out] scalar offset in `destFrame`
     * @param destFrame single frame were all particles are copied in
     * @param srcBox particle box were particles are read from
     * @param particleFilter filter to select particles
     * @param localOffset Offset of the local domain, used to calculate globalCellIdx,
     *                    can be negative for the first GPU: localDomain.offset - globalWindow.offset
     * @param mapper mapper which describes the area where particles are copied from
     */
    template<class T_DestFrame, class T_SrcBox, class T_Filter, class T_Space, class T_Mapping>
    void operator()(int& counter, T_DestFrame destFrame, T_SrcBox srcBox, T_Filter particleFilter, T_Space localOffset, T_Mapping mapper)
    {
        #pragma omp parallel for
        for (int linearBlockIdx = 0;
             linearBlockIdx < m_gridDim.productOfComponents();
             ++linearBlockIdx
             )
        {
            // local copy for each omp thread
            T_Filter filter = particleFilter;
            DataSpace<T_dim> blockIdx(DataSpaceOperations<T_dim>::map(m_gridDim, linearBlockIdx));

            using namespace PMacc::particles::operations;

            typedef T_DestFrame DestFrameType;
            typedef typename T_SrcBox::FrameType SrcFrameType;
            typedef typename T_SrcBox::FramePtr SrcFramePtr;

            typedef T_Mapping Mapping;
            typedef typename Mapping::SuperCellSize SuperCellSize;


            const int particlesPerFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;
            int localIdxs[particlesPerFrame];

            const DataSpace<Mapping::Dim> superCellIdx = mapper.getSuperCellIndex(blockIdx);
            const DataSpace<Mapping::Dim> superCellPosition((superCellIdx - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());
            filter.setSuperCellPosition(superCellPosition);

            SrcFramePtr srcFramePtr = srcBox.getFirstFrame(superCellIdx);

            /* Loop over all frames in current super cell */
            while (srcFramePtr.isValid())
            {
                /* Count number of particles in current frame and init its indices */
                int curNumParticles = 0;
                for (int threadIdx = 0; threadIdx < particlesPerFrame; ++threadIdx)
                {
                    PMACC_AUTO(parSrc, (srcFramePtr[threadIdx]));
                    /* Check if particle exists and is not filtered */
                    if (parSrc[multiMask_] == 1 && filter(*srcFramePtr, threadIdx))
                        localIdxs[threadIdx] = curNumParticles++;
                    else
                        localIdxs[threadIdx] = -1;
                }

                int globalOffset;
                /* atomic update with openmp*/
                #pragma omp critical
                {
                    globalOffset = counter;
                    counter += curNumParticles;
                }

                for (int threadIdx = 0; threadIdx < particlesPerFrame; ++threadIdx)
                {
                    if (localIdxs[threadIdx] != -1)
                    {
                        PMACC_AUTO(parSrc, (srcFramePtr[threadIdx]));
                        PMACC_AUTO(parDest, destFrame[globalOffset + localIdxs[threadIdx]]);
                        PMACC_AUTO(parDestNoGlobalIdx, deselect<globalCellIdx<> >(parDest));
                        assign(parDestNoGlobalIdx, parSrc);
                        /*calculate global cell index*/
                        DataSpace<Mapping::Dim> localCellIdx(DataSpaceOperations<Mapping::Dim>::template map<SuperCellSize>(parSrc[localCellIdx_]));
                        parDest[globalCellIdx_] = localOffset + superCellPosition + localCellIdx;
                    }
                }
                /*get next frame in supercell*/
                srcFramePtr = srcBox.getNextFrame(srcFramePtr);

            }
        }
    }

};

} //namespace operations
} //namespace particles
} //namespace PMacc
