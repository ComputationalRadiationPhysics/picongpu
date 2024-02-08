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
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <array>
#include <cstddef>
#include <cstdio>


namespace picongpu::particles::collision
{
    template<bool useScreeningLength>
    struct InterCollision
    {
        HINLINE InterCollision()
        {
            constexpr auto numScreeningSpecies
                = pmacc::mp_size<picongpu::particles::collision::CollisionScreeningSpecies>::value;
            PMACC_CASSERT_MSG(
                _CollisionScreeningSpecies_can_not_be_empty_when_dynamic_coulomb_log_is_used,
                (useScreeningLength && numScreeningSpecies > 0u) || !useScreeningLength);
            if constexpr(useScreeningLength)
            {
                constexpr uint32_t slot = screeningLengthSlot;
                DataConnector& dc = Environment<>::get().DataConnector();
                auto field = dc.get<FieldTmp>(FieldTmp::getUniqueId(slot));
                screeningLengthSquared = (field->getGridBuffer().getDeviceBuffer().getDataBox());
            }
        }

    private:
        PMACC_ALIGN(screeningLengthSquared, FieldTmp::DataBoxType);
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

    public:
        template<
            typename T_ParBox0,
            typename T_ParBox1,
            typename T_Mapping,
            typename T_Worker,
            typename T_DeviceHeapHandle,
            typename T_RngHandle,
            typename T_SrcCollisionFunctor,
            typename T_Filter0,
            typename T_Filter1,
            typename T_SumCoulombLogBox,
            typename T_SumSParamBox,
            typename T_TimesCollidedBox>
        DINLINE void operator()(
            T_Worker const& worker,
            T_ParBox0 pb0,
            T_ParBox1 pb1,
            T_Mapping const mapper,
            T_DeviceHeapHandle deviceHeapHandle,
            T_RngHandle rngHandle,
            T_SrcCollisionFunctor const srcCollisionFunctor,
            T_Filter0 filter0,
            T_Filter1 filter1,
            T_SumCoulombLogBox sumCoulombLogBox,
            T_SumSParamBox sumSParamBox,
            T_TimesCollidedBox timesCollidedBox) const
        {
            using namespace pmacc::particles::operations;

            constexpr uint32_t numCellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

            using Frame0 = typename T_ParBox0::FrameType;
            using Frame1 = typename T_ParBox1::FrameType;

            PMACC_SMEM(worker, nppc, memory::Array<uint32_t, numCellsPerSuperCell>);

            PMACC_SMEM(worker, parCellList0, detail::ListEntry<T_ParBox0, numCellsPerSuperCell>);
            PMACC_SMEM(worker, parCellList1, detail::ListEntry<T_ParBox1, numCellsPerSuperCell>);
            PMACC_SMEM(worker, densityArray0, memory::Array<float_X, numCellsPerSuperCell>);
            PMACC_SMEM(worker, densityArray1, memory::Array<float_X, numCellsPerSuperCell>);

            constexpr bool ifAverageLog = !std::is_same<T_SumCoulombLogBox, std::nullptr_t>::value;
            constexpr bool ifAverageSParam = !std::is_same<T_SumSParamBox, std::nullptr_t>::value;
            constexpr bool ifTimesCollided = !std::is_same<T_TimesCollidedBox, std::nullptr_t>::value;
            constexpr bool ifDebug = ifAverageLog && ifAverageSParam && ifTimesCollided;

            DataSpace<simDim> const superCellIdx
                = mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(worker.getAcc())));

            // offset of the superCell (in cells, without any guards) to the
            // origin of the local domain
            DataSpace<simDim> const localSuperCellOffset = superCellIdx - mapper.getGuardingSuperCells();
            auto rngOffset = DataSpace<simDim>::create(0);
            rngOffset.x() = worker.getWorkerIdx();
            auto numRNGsPerSuperCell = DataSpace<simDim>::create(1);
            numRNGsPerSuperCell.x() = numFrameSlots;
            rngHandle.init(localSuperCellOffset * numRNGsPerSuperCell + rngOffset);

            auto accFilter0 = filter0(worker, localSuperCellOffset);
            auto accFilter1 = filter1(worker, localSuperCellOffset);

            auto forEachCell = lockstep::makeForEach<numCellsPerSuperCell>(worker);

            prepareList(worker, forEachCell, pb0, superCellIdx, deviceHeapHandle, parCellList0, nppc, accFilter0);

            prepareList(worker, forEachCell, pb1, superCellIdx, deviceHeapHandle, parCellList1, nppc, accFilter1);

            using FramePtr0 = typename T_ParBox0::FramePtr;
            using FramePtr1 = typename T_ParBox1::FramePtr;
            detail::cellDensity<FramePtr0>(worker, forEachCell, parCellList0, densityArray0, accFilter0);
            detail::cellDensity<FramePtr1>(worker, forEachCell, parCellList1, densityArray1, accFilter1);
            worker.sync();

            // shuffle indices list of the longest particle list
            forEachCell(
                [&](uint32_t const linearIdx)
                {
                    uint32_t maxListLength = math::max(parCellList0.size(linearIdx), parCellList1.size(linearIdx));

                    uint32_t* parIdListLong = parCellList0.size(linearIdx) == maxListLength
                        ? parCellList0.particleIds(linearIdx)
                        : parCellList1.particleIds(linearIdx);
                    detail::shuffle(worker, parIdListLong, maxListLength, rngHandle);
                });

            auto collisionFunctorCtx = forEachCell(
                [&](uint32_t const linearIdx)
                {
                    return inCellCollisions(
                        worker,
                        rngHandle,
                        srcCollisionFunctor,
                        localSuperCellOffset,
                        superCellIdx,
                        densityArray0[linearIdx],
                        densityArray1[linearIdx],
                        parCellList0.getParticlesAccessor(linearIdx),
                        parCellList1.getParticlesAccessor(linearIdx),
                        linearIdx);
                });

            worker.sync();

            parCellList0.finalize(worker, deviceHeapHandle);
            parCellList1.finalize(worker, deviceHeapHandle);

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
                    [&](uint32_t idx, auto const& collisionFunctor)
                    {
                        const auto timesUsed = static_cast<uint64_t>(collisionFunctor.timesUsed);
                        if(timesUsed > 0u)
                        {
                            cupla::atomicAdd(
                                worker.getAcc(),
                                &sumCoulombLogBlock,
                                static_cast<float_X>(collisionFunctor.sumCoulombLog),
                                ::alpaka::hierarchy::Threads{});
                            cupla::atomicAdd(
                                worker.getAcc(),
                                &sumSParamBlock,
                                static_cast<float_X>(collisionFunctor.sumSParam),
                                ::alpaka::hierarchy::Threads{});
                            cupla::atomicAdd(

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
                        cupla::atomicAdd(
                            worker.getAcc(),
                            &(sumCoulombLogBox[0]),
                            sumCoulombLogBlock,
                            ::alpaka::hierarchy::Blocks{});
                        cupla::atomicAdd(
                            worker.getAcc(),
                            &(sumSParamBox[0]),
                            sumSParamBlock,
                            ::alpaka::hierarchy::Blocks{});
                        cupla::atomicAdd(
                            worker.getAcc(),
                            &(timesCollidedBox[0]),
                            timesCollidedBlock,
                            ::alpaka::hierarchy::Blocks{});
                    });
            }
        }


        template<
            typename T_Worker,
            typename T_RngHandle,
            typename T_SrcCollisionFunctor,
            typename T_ParAccessor0,
            typename T_ParAccessor1>
        DINLINE decltype(auto) inCellCollisions(
            T_Worker const& worker,
            T_RngHandle& rngHandle,
            T_SrcCollisionFunctor const& srcCollisionFunctor,
            DataSpace<simDim> const& localSuperCellOffset,
            DataSpace<simDim> const& superCellIdx,
            float_X const& density0,
            float_X const& density1,
            T_ParAccessor0 const& parAccessor0,
            T_ParAccessor1 const& parAccessor1,
            [[maybe_unused]] int32_t linearCellIdx) const
        {
            uint32_t const size0 = parAccessor0.size();
            uint32_t const size1 = parAccessor1.size();
            uint32_t const minListLength = math::min(size0, size1);
            uint32_t const maxListLength = math::max(size0, size1);

            auto destCollisionFunctor
                = srcCollisionFunctor(worker, localSuperCellOffset, density0, density1, maxListLength);

            if constexpr(useScreeningLength)
            {
                auto const shifted = screeningLengthSquared.shift(superCellIdx * SuperCellSize::toRT());
                auto const idxInSuperCell = pmacc::math::mapToND(SuperCellSize::toRT(), linearCellIdx);
                destCollisionFunctor.coulombLogFunctor.screeningLengthSquared_m = shifted(idxInSuperCell)[0];
            }

            if(minListLength != 0u)
            {
                for(uint32_t i = 0; i < maxListLength; ++i)
                {
                    auto par0 = parAccessor0[i % size0];
                    auto par1 = parAccessor1[i % size1];
                    destCollisionFunctor.duplicationCorrection
                        = duplicationCorrection(i, minListLength, maxListLength);
                    destCollisionFunctor(detail::makeCollisionContext(worker, rngHandle), par0, par1);
                }
            }
            return destCollisionFunctor;
        }
    };

    /* Run kernel for collisions between two species.
     *
     * @tparam T_CollisionFunctor A binary particle functor defining a single macro particle collision in
     * the binary-collision algorithm.
     * @tparam T_FilterPair A pair of particle filters, each for each species
     *     in the colliding pair.
     * @tparam T_Species0 1st colliding species.
     * @tparam T_Species1 2nd colliding species.
     */
    template<
        typename T_CollisionFunctor,
        typename T_FilterPair,
        typename T_Species0,
        typename T_Species1,
        uint32_t colliderId,
        uint32_t pairId>
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
            auto species0 = dc.get<Species0>(FrameType0::getName());
            auto species1 = dc.get<Species1>(FrameType1::getName());

            // Use mapping information from the first species:
            auto const mapper = makeAreaMapper<CORE + BORDER>(species0->getCellDescription());

            //! random number generator
            using RNGFactory = pmacc::random::RNGProvider<simDim, random::Generator>;
            using Kernel = typename CollisionFunctor::CallingInterKernel;
            auto workerCfg = lockstep::makeWorkerCfg<FrameType0::frameSize>();

            constexpr bool ifDebug = CollisionFunctor::ifDebug_m;
            if constexpr(ifDebug)
            {
                GridBuffer<float_X, DIM1> sumCoulombLog(DataSpace<DIM1>(1));
                sumCoulombLog.getDeviceBuffer().setValue(0.0_X);
                GridBuffer<float_X, DIM1> sumSParam(DataSpace<DIM1>(1));
                sumSParam.getDeviceBuffer().setValue(0.0_X);
                GridBuffer<uint64_t, DIM1> timesCollided(DataSpace<DIM1>(1));
                timesCollided.getDeviceBuffer().setValue(0u);
                PMACC_LOCKSTEP_KERNEL(Kernel{}, workerCfg)
                (mapper.getGridDim())(
                    species0->getDeviceParticlesBox(),
                    species1->getDeviceParticlesBox(),
                    mapper,
                    deviceHeap->getAllocatorHandle(),
                    RNGFactory::createHandle(),
                    CollisionFunctor(currentStep),
                    particles::filter::IUnary<Filter0>{currentStep},
                    particles::filter::IUnary<Filter1>{currentStep},
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
                PMACC_LOCKSTEP_KERNEL(Kernel{}, workerCfg)
                (mapper.getGridDim())(
                    species0->getDeviceParticlesBox(),
                    species1->getDeviceParticlesBox(),
                    mapper,
                    deviceHeap->getAllocatorHandle(),
                    RNGFactory::createHandle(),
                    CollisionFunctor(currentStep),
                    particles::filter::IUnary<Filter0>{currentStep},
                    particles::filter::IUnary<Filter1>{currentStep},
                    nullptr,
                    nullptr,
                    nullptr);
            }
        }
    };
} // namespace picongpu::particles::collision
