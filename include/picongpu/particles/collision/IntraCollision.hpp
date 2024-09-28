/* Copyright 2019-2023 Rene Widera, Pawel Ordyna
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/collision/detail/CollisionContext.hpp"
#include "picongpu/particles/collision/detail/ListEntry.hpp"
#include "picongpu/particles/collision/detail/cellDensity.hpp"
#include "picongpu/particles/collision/fieldSlots.hpp"

#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstddef>
#include <cstdio>

namespace picongpu::particles::collision
{
    template<bool useScreeningLength>
    struct IntraCollision
    {
        HINLINE IntraCollision()
        {
            if constexpr(useScreeningLength)
            {
                constexpr uint32_t slot = screeningLengthSlot;
                DataConnector& dc = Environment<>::get().DataConnector();
                auto field = dc.get<FieldTmp>(FieldTmp::getUniqueId(slot));
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
            typename T_Worker,
            typename T_DeviceHeapHandle,
            typename T_RngHandle,
            typename T_SrcCollisionFunctor,
            typename T_Filter,
            typename T_SumCoulombLogBox,
            typename T_SumSParamBox,
            typename T_TimesCollidedBox>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox pb,
            T_Mapping const mapper,
            T_DeviceHeapHandle deviceHeapHandle,
            T_RngHandle rngHandle,
            T_SrcCollisionFunctor const srcCollisionFunctor,
            T_Filter filter,
            T_SumCoulombLogBox sumCoulombLogBox,
            T_SumSParamBox sumSParamBox,
            T_TimesCollidedBox timesCollidedBox) const
        {
            using namespace pmacc::particles::operations;

            constexpr uint32_t numCellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            PMACC_SMEM(worker, nppc, memory::Array<uint32_t, numCellsPerSuperCell>);
            PMACC_SMEM(worker, parCellList, detail::ListEntry<T_ParBox, numCellsPerSuperCell>);
            PMACC_SMEM(worker, densityArray, memory::Array<float_X, numCellsPerSuperCell>);

            constexpr bool ifAverageLog = !std::is_same<T_SumCoulombLogBox, std::nullptr_t>::value;
            constexpr bool ifAverageSParam = !std::is_same<T_SumSParamBox, std::nullptr_t>::value;
            constexpr bool ifTimesCollided = !std::is_same<T_TimesCollidedBox, std::nullptr_t>::value;
            constexpr bool ifDebug = ifAverageLog && ifAverageSParam && ifTimesCollided;
            DataSpace<simDim> const superCellIdx = mapper.getSuperCellIndex(worker.blockDomIdxND());

            auto& superCell = pb.getSuperCell(superCellIdx);
            uint32_t numParticles = superCell.getNumParticles();

            // if we do not have particles there is no need to perform any calculations
            if(numParticles <= 1u)
                return;

            // offset of the superCell (in cells, without any guards) to the
            // origin of the local domain
            DataSpace<simDim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();
            auto rngOffset = DataSpace<simDim>::create(0);
            rngOffset.x() = worker.workerIdx();
            auto numRNGsPerSuperCell = DataSpace<simDim>::create(1);
            numRNGsPerSuperCell.x() = numFrameSlots;
            rngHandle.init(localSuperCellOffset * numRNGsPerSuperCell + rngOffset);

            auto accFilter = filter(worker, localSuperCellOffset);

            auto forEachCell = lockstep::makeForEach<numCellsPerSuperCell>(worker);

            prepareList(worker, forEachCell, pb, superCellIdx, deviceHeapHandle, parCellList, nppc, accFilter);

            using FramePtr = typename T_ParBox::FramePtr;
            detail::cellDensity<FramePtr>(worker, forEachCell, parCellList, densityArray, accFilter);

            worker.sync();

            // shuffle indices list
            forEachCell(
                [&](uint32_t const linearIdx) {
                    detail::shuffle(
                        worker,
                        parCellList.particleIds(linearIdx),
                        parCellList.size(linearIdx),
                        rngHandle);
                });

            auto collisionFunctorCtx = forEachCell(
                [&](int32_t const idx)
                {
                    auto parAccess = parCellList.getParticlesAccessor(idx);
                    uint32_t const sizeAll = parAccess.size();
                    uint32_t potentialPartners = sizeAll - 1u + sizeAll % 2u;
                    auto collisionFunctor = srcCollisionFunctor(
                        worker,
                        localSuperCellOffset,
                        densityArray[idx],
                        densityArray[idx],
                        potentialPartners);
                    if(sizeAll >= 2u)
                    {
                        if constexpr(useScreeningLength)
                        {
                            auto const shifted = screeningLengthSquared.shift(superCellIdx * SuperCellSize::toRT());
                            auto const idxInSuperCell = pmacc::math::mapToND(SuperCellSize::toRT(), idx);
                            collisionFunctor.coulombLogFunctor.screeningLengthSquared_m = shifted(idxInSuperCell)[0];
                        }
                        for(uint32_t i = 0; i < sizeAll; i += 2)
                        {
                            auto parEven = parAccess[i];
                            auto parOdd = parAccess[(i + 1) % sizeAll];
                            // In Higginson 2020 eq. (31) s has an additional 1/2 factor for
                            // intraCollisions (compare with 29 and use m_aa = 1/2 m_a). But this seems to be a
                            // typo smilei doesn't include this extra factor. It was applied here in the
                            // previous version, via the duplication correction.

                            collisionFunctor.duplicationCorrection = duplicationCorrection(i, sizeAll);
                            collisionFunctor(detail::makeCollisionContext(worker, rngHandle), parEven, parOdd);
                        }
                    }

                    return collisionFunctor;
                });

            worker.sync();

            parCellList.finalize(worker, deviceHeapHandle);

            if constexpr(ifDebug)
            {
                auto onlyMaster = lockstep::makeMaster(worker);

                PMACC_SMEM(worker, sumCoulombLogBlock, float_X);
                PMACC_SMEM(worker, sumSParamBlock, float_X);
                PMACC_SMEM(worker, timesCollidedBlock, uint64_t);
                onlyMaster(
                    [&]()
                    {
                        sumCoulombLogBlock = 0.0_X;
                        sumSParamBlock = 0.0_X;
                        timesCollidedBlock = 0.0_X;
                    });
                worker.sync();
                forEachCell(
                    [&](uint32_t const idx, auto const& collisionFunctor)
                    {
                        auto timesUsed = static_cast<uint64_t>(collisionFunctor.timesUsed);
                        if(timesUsed > 0u)
                        {
                            alpaka::atomicAdd(
                                worker.getAcc(),
                                &sumCoulombLogBlock,
                                static_cast<float_X>(collisionFunctor.sumCoulombLog),
                                ::alpaka::hierarchy::Threads{});
                            alpaka::atomicAdd(
                                worker.getAcc(),
                                &sumSParamBlock,
                                static_cast<float_X>(collisionFunctor.sumSParam),
                                ::alpaka::hierarchy::Threads{});
                            alpaka::atomicAdd(
                                worker.getAcc(),
                                &timesCollidedBlock,
                                timesUsed,
                                ::alpaka::hierarchy::Threads{});
                        }
                    },
                    collisionFunctorCtx);

                worker.sync();

                onlyMaster(
                    [&]()
                    {
                        alpaka::atomicAdd(
                            worker.getAcc(),
                            &(sumCoulombLogBox[0]),
                            sumCoulombLogBlock,
                            ::alpaka::hierarchy::Blocks{});
                        alpaka::atomicAdd(
                            worker.getAcc(),
                            &(sumSParamBox[0]),
                            sumSParamBlock,
                            ::alpaka::hierarchy::Blocks{});
                        alpaka::atomicAdd(
                            worker.getAcc(),
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
    template<typename T_CollisionFunctor, typename T_Filter, typename T_Species, uint32_t colliderId, uint32_t pairId>
    struct DoIntraCollision<T_CollisionFunctor, FilterPair<T_Filter, T_Filter>, T_Species, colliderId, pairId>
    {
        /* Run kernel
         *
         * @param deviceHeap A pointer to device heap for allocating particle lists.
         * @param currentStep The current simulation step.
         */
        void operator()(const std::shared_ptr<DeviceHeap>& deviceHeap, uint32_t currentStep, IdGenerator idGen)
        {
            using Species = T_Species;
            using FrameType = typename Species::FrameType;
            using Filter = typename T_Filter::template apply<Species>::type;
            using CollisionFunctor = T_CollisionFunctor;

            DataConnector& dc = Environment<>::get().DataConnector();
            auto species = dc.get<Species>(FrameType::getName());

            auto const mapper = makeAreaMapper<CORE + BORDER>(species->getCellDescription());

            using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
            using Kernel = typename CollisionFunctor::CallingIntraKernel;

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
                PMACC_LOCKSTEP_KERNEL(Kernel{}).config(mapper.getGridDim(), *species)(
                    species->getDeviceParticlesBox(),
                    mapper,
                    deviceHeap->getAllocatorHandle(),
                    RNGFactory::createHandle(),
                    CollisionFunctor(currentStep),
                    particles::filter::IUnary<Filter>{currentStep, idGen},
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
                    sumCoulombLog.getHostBuffer().data(),
                    1,
                    mpi::reduceMethods::Reduce());
                reduce(
                    pmacc::math::operation::Add(),
                    &reducedSParam,
                    sumSParam.getHostBuffer().data(),
                    1,
                    mpi::reduceMethods::Reduce());
                reduce(
                    pmacc::math::operation::Add(),
                    &reducedTimesCollided,
                    timesCollided.getHostBuffer().data(),
                    1,
                    mpi::reduceMethods::Reduce());

                if(reduce.hasResult(mpi::reduceMethods::Reduce()))
                {
                    std::ofstream outFile{};
                    std::string fileName = "debug_values_collider_" + std::to_string(colliderId) + "_species_pair_"
                        + std::to_string(pairId) + ".dat";
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
                PMACC_LOCKSTEP_KERNEL(Kernel{}).config(mapper.getGridDim(), *species)(
                    species->getDeviceParticlesBox(),
                    mapper,
                    deviceHeap->getAllocatorHandle(),
                    RNGFactory::createHandle(),
                    CollisionFunctor(currentStep),
                    particles::filter::IUnary<Filter>{currentStep, idGen},
                    nullptr,
                    nullptr,
                    nullptr);
            }
        }
    };

} // namespace picongpu::particles::collision
