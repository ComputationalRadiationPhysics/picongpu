/* Copyright 2019-2022 Rene Widera, Pawel Ordyna
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
#include "picongpu/particles/collision/fieldSlots.hpp"

#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstddef>
#include <cstdio>

namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            template<uint32_t T_numWorkers, bool useScreeningLength>
            struct IntraCollision
            {
                HINLINE IntraCollision()
                {
                    if constexpr(useScreeningLength)
                    {
                        constexpr uint32_t slot = screeningLengthSlot;
                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto field = dc.get<FieldTmp>(FieldTmp::getUniqueId(slot), true);
                        screeningLengthSquared = field->getGridBuffer().getDeviceBuffer().getDataBox();
                    }
                }

            private:
                PMACC_ALIGN(screeningLengthSquared, FieldTmp::DataBoxType);
                /* Get the duplication correction for a collision
                 *
                 * A particle duplication is how many times a particle collides in the current time step.
                 * The duplication correction is equal to max(D_0, D_1) where D_0, D_1 are the duplications
                 * of the colliding particles.
                 * In a case of internal collisions all particles collide once expect for the 1st one if the total
                 * number of particles is odd.
                 *
                 * @param idx the index of the particle in the particle list of the cell
                 * @param sizeAll the length of the particle list of the cell
                 */
                DINLINE static uint32_t duplicationCorrection(uint32_t const idx, uint32_t const sizeAll)
                {
                    // All particle collide once.
                    if(sizeAll % 2u == 0u)
                        return 1u;
                    // The first particle collides twice and the last one collides with the first one.
                    // for the rest the correction is 1.
                    else
                        return (idx == 0u || idx == sizeAll - 1u) ? 2u : 1u;
                }

            public:
                template<
                    typename T_ParBox,
                    typename T_Mapping,
                    typename T_Acc,
                    typename T_DeviceHeapHandle,
                    typename T_RngHandle,
                    typename T_CollisionFunctor,
                    typename T_Filter,
                    typename T_SumCoulombLogBox,
                    typename T_SumSParamBox,
                    typename T_TimesCollidedBox>
                DINLINE void operator()(
                    T_Acc const& acc,
                    T_ParBox pb,
                    T_Mapping const mapper,
                    T_DeviceHeapHandle deviceHeapHandle,
                    T_RngHandle rngHandle,
                    T_CollisionFunctor const collisionFunctor,
                    T_Filter filter,
                    T_SumCoulombLogBox sumCoulombLogBox,
                    T_SumSParamBox sumSParamBox,
                    T_TimesCollidedBox timesCollidedBox) const
                {
                    using namespace pmacc::particles::operations;

                    using SuperCellSize = typename T_ParBox::FrameType::SuperCellSize;
                    constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                    constexpr uint32_t numWorkers = T_numWorkers;

                    using FramePtr = typename T_ParBox::FramePtr;

                    PMACC_SMEM(acc, nppc, memory::Array<uint32_t, frameSize>);

                    PMACC_SMEM(acc, parCellList, memory::Array<detail::ListEntry, frameSize>);

                    PMACC_SMEM(acc, densityArray, memory::Array<float_X, frameSize>);

                    uint32_t const workerIdx = cupla::threadIdx(acc).x;
                    auto onlyMaster = lockstep::makeMaster(workerIdx);

                    constexpr bool ifAverageLog = !std::is_same<T_SumCoulombLogBox, std::nullptr_t>::value;
                    constexpr bool ifAverageSParam = !std::is_same<T_SumSParamBox, std::nullptr_t>::value;
                    constexpr bool ifTimesCollided = !std::is_same<T_TimesCollidedBox, std::nullptr_t>::value;
                    constexpr bool ifDebug = ifAverageLog && ifAverageSParam && ifTimesCollided;

                    DataSpace<simDim> const superCellIdx
                        = mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc)));

                    // offset of the superCell (in cells, without any guards) to the
                    // origin of the local domain
                    DataSpace<simDim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();
                    rngHandle.init(
                        localSuperCellOffset * SuperCellSize::toRT()
                        + DataSpaceOperations<simDim>::template map<SuperCellSize>(workerIdx));

                    auto& superCell = pb.getSuperCell(superCellIdx);
                    uint32_t numParticlesInSupercell = superCell.getNumParticles();

                    auto accFilter = filter(acc, localSuperCellOffset, lockstep::Worker<T_numWorkers>{workerIdx});

                    /* loop over all particles in the frame */
                    auto forEachFrameElem = lockstep::makeForEach<frameSize, numWorkers>(workerIdx);
                    FramePtr firstFrame = pb.getFirstFrame(superCellIdx);

                    prepareList(
                        acc,
                        forEachFrameElem,
                        deviceHeapHandle,
                        pb,
                        firstFrame,
                        numParticlesInSupercell,
                        parCellList,
                        nppc,
                        accFilter);


                    cellDensity(acc, forEachFrameElem, firstFrame, pb, parCellList, densityArray, accFilter);
                    cupla::__syncthreads(acc);

                    // shuffle indices list
                    forEachFrameElem([&](uint32_t const linearIdx)
                                     { parCellList[linearIdx].shuffle(acc, rngHandle); });

                    auto collisionFunctorCtx = lockstep::makeVar<decltype(collisionFunctor(
                        acc,
                        alpaka::core::declval<DataSpace<simDim> const>(),
                        /*frameSize is used because each virtual worker
                         * is creating **exactly one** functor
                         */
                        alpaka::core::declval<lockstep::Worker<frameSize> const>(),
                        alpaka::core::declval<float_X const>(),
                        alpaka::core::declval<float_X const>(),
                        alpaka::core::declval<uint32_t const>()))>(forEachFrameElem);

                    forEachFrameElem(
                        [&](lockstep::Idx const idx)
                        {
                            uint32_t const sizeAll = parCellList[idx].size;
                            if(sizeAll < 2u)
                                return;
                            // skip particle offset counter
                            uint32_t* listAll = parCellList[idx].ptrToIndicies;
                            uint32_t potentialPartners = sizeAll - 1u + sizeAll % 2u;
                            collisionFunctorCtx[idx] = collisionFunctor(
                                acc,
                                localSuperCellOffset,
                                lockstep::Worker<T_numWorkers>{workerIdx},
                                densityArray[idx],
                                densityArray[idx],
                                potentialPartners);
                            if constexpr(useScreeningLength)
                            {
                                auto const shifted
                                    = screeningLengthSquared.shift(superCellIdx * SuperCellSize::toRT());
                                auto const idxInSuperCell
                                    = DataSpaceOperations<simDim>::template map<SuperCellSize>(idx);
                                collisionFunctorCtx[idx].coulombLogFunctor.screeningLengthSquared_m
                                    = shifted(idxInSuperCell)[0];
                            }
                            for(uint32_t i = 0; i < sizeAll; i += 2)
                            {
                                auto parEven = detail::getParticle(pb, firstFrame, listAll[i]);
                                auto parOdd = detail::getParticle(pb, firstFrame, listAll[(i + 1) % sizeAll]);
                                // In Higginson 2020 eq. (31) s has an additional 1/2 factor for
                                // intraCollisions (compare with 29 and use m_aa = 1/2 m_a). Bu this seams to be a typo
                                // smilei doesn't include this extra factor. It was applied here in the previous
                                // version, via the duplication correction.

                                collisionFunctorCtx[idx].duplicationCorrection = duplicationCorrection(i, sizeAll);
                                (collisionFunctorCtx[idx])(
                                    detail::makeCollisionContext(acc, rngHandle),
                                    parEven,
                                    parOdd);
                            }
                        });

                    cupla::__syncthreads(acc);

                    forEachFrameElem([&](uint32_t const linearIdx)
                                     { parCellList[linearIdx].finalize(acc, deviceHeapHandle); });
                    if constexpr(ifDebug)
                    {
                        PMACC_SMEM(acc, sumCoulombLogBlock, float_X);
                        PMACC_SMEM(acc, sumSParamBlock, float_X);
                        PMACC_SMEM(acc, timesCollidedBlock, uint64_t);
                        onlyMaster(
                            [&]()
                            {
                                sumCoulombLogBlock = 0.0_X;
                                sumSParamBlock = 0.0_X;
                                timesCollidedBlock = 0.0_X;
                            });
                        cupla::__syncthreads(acc);
                        forEachFrameElem(
                            [&](lockstep::Idx const idx)
                            {
                                auto timesUsed = static_cast<uint64_t>(collisionFunctorCtx[idx].timesUsed);
                                if(timesUsed > 0u)
                                {
                                    cupla::atomicAdd(
                                        acc,
                                        &sumCoulombLogBlock,
                                        static_cast<float_X>(collisionFunctorCtx[idx].sumCoulombLog),
                                        ::alpaka::hierarchy::Threads{});
                                    cupla::atomicAdd(
                                        acc,
                                        &sumSParamBlock,
                                        static_cast<float_X>(collisionFunctorCtx[idx].sumSParam),
                                        ::alpaka::hierarchy::Threads{});
                                    cupla::atomicAdd(
                                        acc,
                                        &timesCollidedBlock,
                                        timesUsed,
                                        ::alpaka::hierarchy::Threads{});
                                }
                            });

                        cupla::__syncthreads(acc);

                        onlyMaster(
                            [&]()
                            {
                                cupla::atomicAdd(
                                    acc,
                                    &(sumCoulombLogBox[0]),
                                    sumCoulombLogBlock,
                                    ::alpaka::hierarchy::Blocks{});
                                cupla::atomicAdd(
                                    acc,
                                    &(sumSParamBox[0]),
                                    sumSParamBlock,
                                    ::alpaka::hierarchy::Blocks{});
                                cupla::atomicAdd(
                                    acc,
                                    &(timesCollidedBox[0]),
                                    timesCollidedBlock,
                                    ::alpaka::hierarchy::Blocks{});
                            });
                    }
                }
            };

            /* Run kernel for internal collisions of one species.
             *
             * @tparam T_CollisionFunctor A binary particle functor defining a single macro particle collision in the
             *     binary-collision algorithm.
             * @tparam T_FilterPair A pair of particle filters, each for each species
             *     in the colliding pair.
             * @tparam T_Species0 Colliding species.
             * @tparam T_Species1 2nd colliding species.
             */
            template<
                typename T_CollisionFunctor,
                typename T_FilterPair,
                typename T_Species,
                uint32_t colliderId,
                uint32_t pairId>
            struct DoIntraCollision;

            // A single template specialization. This ensures that the code won't compile if the FilterPair contains
            // two different filters. That wouldn't make much sense for internal collisions.
            template<
                typename T_CollisionFunctor,
                typename T_Filter,
                typename T_Species,
                uint32_t colliderId,
                uint32_t pairId>
            struct DoIntraCollision<T_CollisionFunctor, FilterPair<T_Filter, T_Filter>, T_Species, colliderId, pairId>
            {
                /* Run kernel
                 *
                 * @param deviceHeap A pointer to device heap for allocating particle lists.
                 * @param currentStep The current simulation step.
                 */
                void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep)
                {
                    using Species = T_Species;
                    using FrameType = typename Species::FrameType;
                    using Filter = typename T_Filter::template apply<Species>::type;
                    using CollisionFunctor = T_CollisionFunctor;

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto species = dc.get<Species>(FrameType::getName(), true);

                    auto const mapper = makeAreaMapper<CORE + BORDER>(species->getCellDescription());

                    constexpr uint32_t numWorkers
                        = pmacc::traits::GetNumWorkers<pmacc::math::CT::volume<SuperCellSize>::type::value>::value;
                    using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
                    using Kernel = typename CollisionFunctor::template CallingIntraKernel<numWorkers>;

                    constexpr bool ifDebug = CollisionFunctor::ifDebug_m;
                    if constexpr(ifDebug)
                    {
                        GridBuffer<float_X, DIM1> sumCoulombLog(DataSpace<DIM1>(1));
                        sumCoulombLog.getDeviceBuffer().setValue(0.0_X);
                        GridBuffer<float_X, DIM1> sumSParam(DataSpace<DIM1>(1));
                        sumSParam.getDeviceBuffer().setValue(0.0_X);
                        GridBuffer<uint64_t, DIM1> timesCollided(DataSpace<DIM1>(1));
                        timesCollided.getDeviceBuffer().setValue(0u);

                        /* random number generator */

                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(), numWorkers)(
                            species->getDeviceParticlesBox(),
                            mapper,
                            deviceHeap->getAllocatorHandle(),
                            RNGFactory::createHandle(),
                            CollisionFunctor(currentStep),
                            particles::filter::IUnary<Filter>{currentStep},
                            sumCoulombLog.getDeviceBuffer().getDataBox(),
                            sumSParam.getDeviceBuffer().getDataBox(),
                            timesCollided.getDeviceBuffer().getDataBox());

                        sumCoulombLog.deviceToHost();
                        sumSParam.deviceToHost();
                        timesCollided.deviceToHost();

                        float_X reducedAverageCoulombLog;
                        float_X reducedSParam;
                        uint64_t reducedTimesCollided;

                        mpi::MPIReduce reduce{};
                        reduce(
                            pmacc::math::operation::Add(),
                            &reducedAverageCoulombLog,
                            sumCoulombLog.getHostBuffer().getBasePointer(),
                            1,
                            mpi::reduceMethods::Reduce());
                        reduce(
                            pmacc::math::operation::Add(),
                            &reducedSParam,
                            sumSParam.getHostBuffer().getBasePointer(),
                            1,
                            mpi::reduceMethods::Reduce());
                        reduce(
                            pmacc::math::operation::Add(),
                            &reducedTimesCollided,
                            timesCollided.getHostBuffer().getBasePointer(),
                            1,
                            mpi::reduceMethods::Reduce());

                        if(reduce.hasResult(mpi::reduceMethods::Reduce()))
                        {
                            std::ofstream outFile{};
                            std::string fileName = "debug_values_collider_" + std::to_string(colliderId)
                                + "_species_pair_" + std::to_string(pairId) + ".dat";
                            outFile.open(fileName.c_str(), std::ofstream::out | std::ostream::app);
                            outFile << currentStep << " "
                                    << reducedAverageCoulombLog / static_cast<float_X>(reducedTimesCollided) << " "
                                    << reducedSParam / static_cast<float_X>(reducedTimesCollided) << std::endl;
                            outFile.flush();
                            outFile.close();
                        }
                    }
                    else
                    {
                        PMACC_KERNEL(Kernel{})
                        (mapper.getGridDim(), numWorkers)(
                            species->getDeviceParticlesBox(),
                            mapper,
                            deviceHeap->getAllocatorHandle(),
                            RNGFactory::createHandle(),
                            CollisionFunctor(currentStep),
                            particles::filter::IUnary<Filter>{currentStep},
                            nullptr,
                            nullptr,
                            nullptr);
                    }
                }
            };

        } // namespace collision
    } // namespace particles
} // namespace picongpu
