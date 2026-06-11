/*
 * SPDX-FileCopyrightText: Rene Widera, Felix Schmitt, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/lockstep/Worker.hpp"
#include "pmacc/particles/Identifier.hpp"
#include "pmacc/particles/operations/Deselect.hpp"
#include "pmacc/types.hpp"

#include "pmacc/math/vector/compile-time/Vector.hpp"

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
            struct ConcatListOfFrames
            {
                ConcatListOfFrames() = default;

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
                    T_Filter const particleFilter,
                    T_Space const domainOffset,
                    T_Identifier const domainCellIdxIdentifier,
                    T_Mapping mapper,
                    T_ParticleFilter& parFilter)
                {
                    auto gridSize = mapper.getGridDim();
#pragma omp parallel for
                    for(int linearBlockIdx = 0; linearBlockIdx < gridSize.productOfComponents(); ++linearBlockIdx)
                    {
                        // local copy for each omp thread
                        T_Filter filter = particleFilter;
                        auto blockIndexND = pmacc::math::mapToND(gridSize, linearBlockIdx);

                        using namespace pmacc::particles::operations;

                        typedef T_DestFrame DestFrameType;
                        typedef typename T_SrcBox::FrameType SrcFrameType;
                        typedef typename T_SrcBox::FramePtr SrcFramePtr;

                        typedef T_Mapping Mapping;
                        typedef typename Mapping::SuperCellSize SuperCellSize;


                        int const particlesPerFrame = T_SrcBox::frameSize;
                        int localIdxs[particlesPerFrame];

                        DataSpace<Mapping::Dim> const superCellIdx = mapper.getSuperCellIndex(blockIndexND);
                        DataSpace<Mapping::Dim> const superCellPosition(
                            (superCellIdx - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());
                        filter.setSuperCellPosition(superCellPosition);
                        auto accParFilter = parFilter(
                            1, /* @todo this is a hack, please add a alpaka accelerator here*/
                            superCellIdx - mapper.getGuardingSuperCells());

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
                                if(parSrc[multiMask_] == 1 && filter(parSrc))
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
                                    parDestNoDomainIdx = parSrc;
                                    /* calculate cell index for user-defined domain */
                                    DataSpace<Mapping::Dim> localCellIdx = pmacc::math::mapToND(
                                        SuperCellSize::toRT(),
                                        static_cast<int>(parSrc[localCellIdx_]));
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
