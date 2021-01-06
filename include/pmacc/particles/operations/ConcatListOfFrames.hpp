/* Copyright 2013-2021 Rene Widera, Felix Schmitt, Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"
#include "pmacc/math/vector/compile-time/Vector.hpp"

#include "pmacc/mappings/threads/WorkerCfg.hpp"

namespace pmacc
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
             * - add new cellIdx attribute relative to a user-defined domain
             */
            template<unsigned T_dim>
            struct ConcatListOfFrames
            {
                DataSpace<T_dim> m_gridSize;

                ConcatListOfFrames(const DataSpace<T_dim>& gridSize) : m_gridSize(gridSize)
                {
                }

                /** concatenate list of frames to single frame
                 *
                 * @param counter[in,out] scalar offset in `destFrame`
                 * @param destFrame single frame were all particles are copied in
                 * @param srcBox particle box were particles are read from
                 * @param particleFilter filter to select particles
                 * @param domainOffset offset to a user-defined domain. Can, e.g. be used to
                 *                     calculate a totalCellIdx, adding
                 *                     globalDomain.offset + localDomain.offset
                 * @param domainCellIdxIdentifier the identifier for the particle cellIdx
                 *                                that is calculated with respect to
                 *                                domainOffset
                 * @param mapper mapper which describes the area where particles are copied from
                 * @param parFilter particle filter method, must fulfill the interface of pmacc::filter::Interface
                 *                  The working domain for the filter is supercells.
                 */
                template<
                    class T_DestFrame,
                    class T_SrcBox,
                    class T_Filter,
                    class T_Space,
                    class T_Identifier,
                    class T_Mapping,
                    typename T_ParticleFilter>
                void operator()(
                    int& counter,
                    T_DestFrame destFrame,
                    T_SrcBox srcBox,
                    const T_Filter particleFilter,
                    const T_Space domainOffset,
                    const T_Identifier domainCellIdxIdentifier,
                    const T_Mapping mapper,
                    T_ParticleFilter& parFilter)
                {
#pragma omp parallel for
                    for(int linearBlockIdx = 0; linearBlockIdx < m_gridSize.productOfComponents(); ++linearBlockIdx)
                    {
                        // local copy for each omp thread
                        T_Filter filter = particleFilter;
                        DataSpace<T_dim> blockIndex(DataSpaceOperations<T_dim>::map(m_gridSize, linearBlockIdx));

                        using namespace pmacc::particles::operations;
                        using namespace mappings::threads;

                        typedef T_DestFrame DestFrameType;
                        typedef typename T_SrcBox::FrameType SrcFrameType;
                        typedef typename T_SrcBox::FramePtr SrcFramePtr;

                        typedef T_Mapping Mapping;
                        typedef typename Mapping::SuperCellSize SuperCellSize;


                        const int particlesPerFrame = pmacc::math::CT::volume<SuperCellSize>::type::value;
                        int localIdxs[particlesPerFrame];

                        const DataSpace<Mapping::Dim> superCellIdx = mapper.getSuperCellIndex(blockIndex);
                        const DataSpace<Mapping::Dim> superCellPosition(
                            (superCellIdx - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());
                        filter.setSuperCellPosition(superCellPosition);
                        auto accParFilter = parFilter(
                            1, /* @todo this is a hack, please add a alpaka accelerator here*/
                            superCellIdx - mapper.getGuardingSuperCells(),
                            WorkerCfg<1>{0} /* @todo this is a workaround because we use no alpaka*/
                        );

                        SrcFramePtr srcFramePtr = srcBox.getFirstFrame(superCellIdx);

                        /* Loop over all frames in current super cell */
                        while(srcFramePtr.isValid())
                        {
                            /* Count number of particles in current frame and init its indices */
                            int curNumParticles = 0;
                            for(int particleIdx = 0; particleIdx < particlesPerFrame; ++particleIdx)
                            {
                                localIdxs[particleIdx] = -1;
                                auto parSrc = (srcFramePtr[particleIdx]);
                                /* Check if particle exists and is not filtered */
                                if(parSrc[multiMask_] == 1 && filter(*srcFramePtr, particleIdx))
                                    if(accParFilter(
                                           1, /* @todo this is a hack, please add a alpaka accelerator here*/
                                           parSrc))
                                        localIdxs[particleIdx] = curNumParticles++;
                            }

                            int globalOffset;
/* atomic update with openmp*/
#pragma omp critical
                            {
                                globalOffset = counter;
                                counter += curNumParticles;
                            }

                            for(int particleIdx = 0; particleIdx < particlesPerFrame; ++particleIdx)
                            {
                                if(localIdxs[particleIdx] != -1)
                                {
                                    auto parSrc = (srcFramePtr[particleIdx]);
                                    auto parDest = destFrame[globalOffset + localIdxs[particleIdx]];
                                    auto parDestNoDomainIdx = deselect<T_Identifier>(parDest);
                                    assign(parDestNoDomainIdx, parSrc);
                                    /* calculate cell index for user-defined domain */
                                    DataSpace<Mapping::Dim> localCellIdx(
                                        DataSpaceOperations<Mapping::Dim>::template map<SuperCellSize>(
                                            parSrc[localCellIdx_]));
                                    parDest[domainCellIdxIdentifier] = domainOffset + superCellPosition + localCellIdx;
                                }
                            }
                            /*get next frame in supercell*/
                            srcFramePtr = srcBox.getNextFrame(srcFramePtr);
                        }
                    }
                }
            };

        } // namespace operations
    } // namespace particles
} // namespace pmacc
