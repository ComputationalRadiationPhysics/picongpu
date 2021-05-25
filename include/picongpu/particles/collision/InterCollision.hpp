/* Copyright 2019-2021 Rene Widera, Pawel Ordyna
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

#include "picongpu/particles/collision/detail/CollisionContext.hpp"
#include "picongpu/particles/collision/detail/ListEntry.hpp"
#include "picongpu/particles/collision/detail/cellDensity.hpp"

#include <pmacc/lockstep.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstdio>


namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            template<uint32_t T_numWorkers>
            struct InterCollision
            {
                /* Get the duplication correction for a collision
                 *
                 * A particle duplication is how many times a particle collides in the current time step.
                 * The duplication correction is equal to max(D_0, D_1) where D_0, D_1 are the duplications
                 * of the colliding particles.
                 * In a case of inter-species collisions the particles in the long list collide always once. So that
                 * the correction is the duplication of the particle from the shorter list.
                 *
                 * @param idx the index of the particle in the longer particle list of the cell
                 * @param sizeShort the size of the shorter particle list of the cell
                 * @param sizeLong the size of the longer particle list of the cell
                 */
                DINLINE static uint32_t duplicationCorrection(
                    uint32_t const idx,
                    uint32_t const sizeShort,
                    uint32_t const sizeLong)
                {
                    uint32_t duplication_correction(1u);
                    if(sizeLong > sizeShort) // no need for duplications when sizeLong = sizeShort
                    {
                        // Taken from Higginson 2020 DOI: 10.1016/j.jcp.2020.109450
                        duplication_correction = sizeLong / sizeShort;
                        uint32_t modulo = sizeLong % sizeShort;
                        if((idx % sizeShort) < modulo)
                            duplication_correction += 1u;
                    };
                    return duplication_correction;
                }

                template<
                    typename T_ParBox0,
                    typename T_ParBox1,
                    typename T_Mapping,
                    typename T_Acc,
                    typename T_DeviceHeapHandle,
                    typename T_RngHandle,
                    typename T_CollisionFunctor,
                    typename T_Filter0,
                    typename T_Filter1>
                DINLINE void operator()(
                    T_Acc const& acc,
                    T_ParBox0 pb0,
                    T_ParBox1 pb1,
                    T_Mapping const mapper,
                    T_DeviceHeapHandle deviceHeapHandle,
                    T_RngHandle rngHandle,
                    T_CollisionFunctor const collisionFunctor,
                    float_X coulombLog,
                    T_Filter0 filter0,
                    T_Filter1 filter1) const
                {
                    using namespace pmacc::particles::operations;

                    using SuperCellSize = typename T_ParBox0::FrameType::SuperCellSize;
                    constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    constexpr uint32_t numWorkers = T_numWorkers;

                    using FramePtr0 = typename T_ParBox0::FramePtr;
                    using FramePtr1 = typename T_ParBox1::FramePtr;

                    PMACC_SMEM(acc, nppc, memory::Array<uint32_t, frameSize>);

                    PMACC_SMEM(acc, parCellList0, memory::Array<detail::ListEntry, frameSize>);
                    PMACC_SMEM(acc, parCellList1, memory::Array<detail::ListEntry, frameSize>);
                    PMACC_SMEM(acc, densityArray0, memory::Array<float_X, frameSize>);
                    PMACC_SMEM(acc, densityArray1, memory::Array<float_X, frameSize>);

                    uint32_t const workerIdx = cupla::threadIdx(acc).x;

                    DataSpace<simDim> const superCellIdx
                        = mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc)));

                    // offset of the superCell (in cells, without any guards) to the
                    // origin of the local domain
                    DataSpace<simDim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();
                    rngHandle.init(
                        localSuperCellOffset * SuperCellSize::toRT()
                        + DataSpaceOperations<simDim>::template map<SuperCellSize>(workerIdx));

                    auto accFilter0 = filter0(acc, localSuperCellOffset, lockstep::Worker<T_numWorkers>{workerIdx});
                    auto accFilter1 = filter1(acc, localSuperCellOffset, lockstep::Worker<T_numWorkers>{workerIdx});

                    auto& superCell0 = pb0.getSuperCell(superCellIdx);
                    uint32_t numParticlesInSupercell0 = superCell0.getNumParticles();

                    auto& superCell1 = pb1.getSuperCell(superCellIdx);
                    uint32_t numParticlesInSupercell1 = superCell1.getNumParticles();

                    /* loop over all particles in the frame */
                    auto forEachFrameElem = lockstep::makeForEach<frameSize, numWorkers>(workerIdx);

                    FramePtr0 firstFrame0 = pb0.getFirstFrame(superCellIdx);
                    prepareList(
                        acc,
                        forEachFrameElem,
                        deviceHeapHandle,
                        pb0,
                        firstFrame0,
                        numParticlesInSupercell0,
                        parCellList0,
                        nppc,
                        accFilter0);

                    FramePtr1 firstFrame1 = pb1.getFirstFrame(superCellIdx);
                    prepareList(
                        acc,
                        forEachFrameElem,
                        deviceHeapHandle,
                        pb1,
                        firstFrame1,
                        numParticlesInSupercell1,
                        parCellList1,
                        nppc,
                        accFilter1);

                    cellDensity(acc, forEachFrameElem, firstFrame0, pb0, parCellList0, densityArray0, accFilter0);
                    cellDensity(acc, forEachFrameElem, firstFrame1, pb1, parCellList1, densityArray1, accFilter1);
                    cupla::__syncthreads(acc);

                    // shuffle indices list of the longest particle list
                    forEachFrameElem([&](uint32_t const linearIdx) {
                        // find longer list
                        auto* longParList = parCellList0[linearIdx].size >= parCellList1[linearIdx].size
                            ? &parCellList0[linearIdx]
                            : &parCellList1[linearIdx];
                        (*longParList).shuffle(acc, rngHandle);
                    });

                    auto collisionFunctorCtx = lockstep::makeVar<decltype(collisionFunctor(
                        acc,
                        alpaka::core::declval<DataSpace<simDim> const>(),
                        /* frameSize is used because each virtual worker
                         * is creating **exactly one** functor
                         */
                        alpaka::core::declval<lockstep::Worker<frameSize> const>(),
                        alpaka::core::declval<float_X const>(),
                        alpaka::core::declval<float_X const>(),
                        alpaka::core::declval<uint32_t const>(),
                        alpaka::core::declval<float_X const>()))>(forEachFrameElem);

                    forEachFrameElem([&](lockstep::Idx const idx) {
                        uint32_t const linearIdx = idx;
                        if(parCellList0[linearIdx].size >= parCellList1[linearIdx].size)
                        {
                            inCellCollisions(
                                acc,
                                rngHandle,
                                collisionFunctor,
                                localSuperCellOffset,
                                workerIdx,
                                densityArray0[linearIdx],
                                densityArray1[linearIdx],
                                parCellList0[linearIdx].ptrToIndicies,
                                parCellList1[linearIdx].ptrToIndicies,
                                parCellList0[linearIdx].size,
                                parCellList1[linearIdx].size,
                                pb0,
                                pb1,
                                firstFrame0,
                                firstFrame1,
                                coulombLog,
                                collisionFunctorCtx,
                                idx);
                        }
                        else
                        {
                            inCellCollisions(
                                acc,
                                rngHandle,
                                collisionFunctor,
                                localSuperCellOffset,
                                workerIdx,
                                densityArray1[linearIdx],
                                densityArray0[linearIdx],
                                parCellList1[linearIdx].ptrToIndicies,
                                parCellList0[linearIdx].ptrToIndicies,
                                parCellList1[linearIdx].size,
                                parCellList0[linearIdx].size,
                                pb1,
                                pb0,
                                firstFrame1,
                                firstFrame0,
                                coulombLog,
                                collisionFunctorCtx,
                                idx);
                        }
                    });

                    cupla::__syncthreads(acc);

                    forEachFrameElem([&](uint32_t const linearIdx) {
                        parCellList0[linearIdx].finalize(acc, deviceHeapHandle);
                        parCellList1[linearIdx].finalize(acc, deviceHeapHandle);
                    });
                }


                template<
                    typename T_Acc,
                    typename T_RngHandle,
                    typename T_CollisionFunctor,
                    typename T_ListLong,
                    typename T_ListShort,
                    typename T_SizeLong,
                    typename T_SizeShort,
                    typename T_PBoxLong,
                    typename T_PBoxShort,
                    typename T_FrameLong,
                    typename T_FrameShort,
                    typename T_CollisionFunctorCtx>
                DINLINE void inCellCollisions(
                    T_Acc const& acc,
                    T_RngHandle& rngHandle,
                    T_CollisionFunctor const& collisionFunctor,
                    DataSpace<simDim> const& localSuperCellOffset,
                    uint32_t const& workerIdx,
                    float_X const& densityLong,
                    float_X const& densityShort,
                    T_ListLong& listLong,
                    T_ListShort& listShort,
                    T_SizeLong const& sizeLong,
                    T_SizeShort const& sizeShort,
                    T_PBoxLong const& pBoxLong,
                    T_PBoxShort const& pBoxShort,
                    T_FrameLong const& frameLong,
                    T_FrameShort const& frameShort,
                    float_X const& coulombLog,
                    T_CollisionFunctorCtx& collisionFunctorCtx,
                    lockstep::Idx idx

                ) const
                {
                    collisionFunctorCtx[idx] = collisionFunctor(
                        acc,
                        localSuperCellOffset,
                        lockstep::Worker<T_numWorkers>{workerIdx},
                        densityLong,
                        densityShort,
                        sizeLong,
                        coulombLog);
                    if(sizeShort == 0u)
                        return;
                    for(uint32_t i = 0; i < sizeLong; ++i)
                    {
                        auto parLong = detail::getParticle(pBoxLong, frameLong, listLong[i]);
                        auto parShort = detail::getParticle(pBoxShort, frameShort, listShort[i % sizeShort]);
                        collisionFunctorCtx[idx].duplicationCorrection = duplicationCorrection(i, sizeShort, sizeLong);
                        (collisionFunctorCtx[idx])(detail::makeCollisionContext(acc, rngHandle), parLong, parShort);
                    }
                }
            };

            /* Run kernel for collisions between two species.
             *
             * @tparam T_CollisionFunctor A binary particle functor defining a single macro particle collision in the
             *     binary-collision algorithm.
             * @tparam T_Params A struct defining `coulombLog` for the collisions.
             * @tparam T_FilterPair A pair of particle filters, each for each species
             *     in the colliding pair.
             * @tparam T_Species0 1st colliding species.
             * @tparam T_Species1 2nd colliding species.
             */
            template<
                typename T_CollisionFunctor,
                typename T_Params,
                typename T_FilterPair,
                typename T_Species0,
                typename T_Species1>
            struct DoInterCollision
            {
                /* Run kernel
                 *
                 * @param deviceHeap A pointer to device heap for allocating particle lists.
                 * @param currentStep The current simulation step.
                 */
                HINLINE void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
                {
                    using Species0 = T_Species0;
                    using FrameType0 = typename Species0::FrameType;
                    using Filter0 = typename T_FilterPair::first ::template apply<Species0>::type;

                    using Species1 = T_Species1;
                    using FrameType1 = typename Species1::FrameType;
                    using Filter1 = typename T_FilterPair::second ::template apply<Species1>::type;

                    using CollisionFunctor = T_CollisionFunctor;

                    // Access particle data:
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto species0 = dc.get<Species0>(FrameType0::getName(), true);
                    auto species1 = dc.get<Species1>(FrameType1::getName(), true);

                    // Use mapping information from the first species:
                    AreaMapping<CORE + BORDER, picongpu::MappingDesc> mapper(species0->getCellDescription());

                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;

                    //! random number generator
                    using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                    constexpr float_X coulombLog = T_Params::coulombLog;

                    PMACC_KERNEL(InterCollision<numWorkers>{})
                    (mapper.getGridDim(), numWorkers)(
                        species0->getDeviceParticlesBox(),
                        species1->getDeviceParticlesBox(),
                        mapper,
                        deviceHeap->getAllocatorHandle(),
                        RNGFactory::createHandle(),
                        CollisionFunctor(currentStep),
                        coulombLog,
                        Filter0(),
                        Filter1());
                }
            };

        } // namespace collision
    } // namespace particles
} // namespace picongpu
