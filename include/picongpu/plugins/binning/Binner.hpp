/* Copyright 2023 Tapish Narwal
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

#include "picongpu/plugins/binning/BinningFunctors.hpp"
#include "picongpu/plugins/binning/DomainInfo.hpp"
#include "picongpu/plugins/binning/WriteHist.hpp"
#include "picongpu/plugins/binning/utility.hpp"


namespace picongpu
{
    namespace plugins::binning
    {
        /**
         * Functor to run on divide for time averaging
         */
        // TODO make this generic? apply a functor on all elements of a databox? take in functor?
        template<uint32_t blockSize, uint32_t nAxes>
        struct ProductKernel
        {
            using result_type = void;

            HINLINE
            ProductKernel()
            {
            }

            template<typename T_Worker, typename T_DataSpace, typename T_DepositedQuantity, typename T_DataBox>
            HDINLINE void operator()(
                const T_Worker& worker,
                const T_DataSpace& extentsDataspace,
                const T_DepositedQuantity factor,
                T_DataBox dataBox) const
            {
                auto blockIdx = cupla::blockIdx(worker.getAcc()).x * blockSize;
                auto forEachElemInDataboxChunk = lockstep::makeForEach<blockSize>(worker);
                forEachElemInDataboxChunk(
                    [&](int32_t const linearIdx)
                    {
                        int32_t const linearTid = blockIdx + linearIdx;
                        if(linearTid < extentsDataspace.productOfComponents())
                            dataBox(DataSpace<1u>{static_cast<int>(linearTid)}) *= factor;
                    });
            }
        };


        /**
         * Functor to do volume normaliztion
         * Factor with picongpu units
         * User needs to deal with the units seperately
         */
        // TODO make this more generic? databox operations with another databox?
        // maybe store the normalization in a buffer rather than computing it at every output timestep (memory vs ops)
        // this style vs having the members as params of operator()
        template<uint32_t blockSize, uint32_t nAxes>
        struct BinNormalizationKernel
        {
            using result_type = void;

            HINLINE
            BinNormalizationKernel()
            {
            }

            // TODO check if type stored in histBox is same as axisKernelTuple Type
            template<typename T_Worker, typename T_DataSpace, typename T_AxisKernelTuple, typename T_DataBox>
            HDINLINE void operator()(
                const T_Worker& worker,
                const T_DataSpace& extentsDataspace,
                const T_AxisKernelTuple& axisKernelTuple,
                T_DataBox histBox) const
            {
                // TODO check normDataBox shape is same as histBox
                auto blockIdx = cupla::blockIdx(worker.getAcc()).x * blockSize;
                auto forEachElemInDataboxChunk = lockstep::makeForEach<blockSize>(worker);
                forEachElemInDataboxChunk(
                    [&](int32_t const linearIdx)
                    {
                        int32_t const linearTid = blockIdx + linearIdx;
                        if(linearTid < extentsDataspace.productOfComponents())
                        {
                            pmacc::DataSpace<nAxes> const nDIdx
                                = pmacc::DataSpaceOperations<nAxes>::map(extentsDataspace, linearTid);
                            float_X factor = 1.;
                            apply(
                                [&](auto const&... tupleArgs)
                                {
                                    // uses bin width for axes without dimensions as well should those be ignored?
                                    uint32_t i = 0;
                                    ((factor *= tupleArgs.getBinWidth(nDIdx[i++])), ...);
                                },
                                axisKernelTuple);

                            histBox(linearTid) *= 1 / factor;
                        }
                    });
            }
        };

        void dimensionSubtraction(std::array<double, 7>& outputDims, const std::array<double, 7>& axisDims)
        {
            for(size_t i = 0; i < 7; i++)
            {
                outputDims[i] += -axisDims[i];
            }
        }

        template<typename TBinningData>
        class Binner : public INotify
        {
            using TDepositedQuantity = typename TBinningData::DepositedQuantityType;

        private:
            TBinningData binningData;
            MappingDesc* cellDescription;
            std::unique_ptr<HostDeviceBuffer<TDepositedQuantity, 1>> histBuffer;
            uint32_t accumulateCounter = 0;
            mpi::MPIReduce reduce{};
            bool isMaster = false;
            WriteHist histWriter;

        public:
            Binner(TBinningData bd, MappingDesc* cellDesc) : binningData{bd}, cellDescription{cellDesc}
            {
                Environment<>::get().PluginConnector().setNotificationPeriod(this, binningData.notifyPeriod);

                /**
                 * Allocate and manage global histogram memory here, to facilitate time averaging
                 */

                this->histBuffer = std::make_unique<HostDeviceBuffer<TDepositedQuantity, 1>>(
                    binningData.axisExtentsND.productOfComponents()); // TODO for auto n_bins. allocate full size
                                                                      // buffer here. dont init axisExtents yet

                this->histBuffer->getDeviceBuffer().setValue(TDepositedQuantity(0.0));
            }

            ~Binner() override = default;

            void notify(uint32_t currentStep) override
            {
                // TODO auto range init. Init ranges and AxisKernels
                std::apply([](auto&... tupleArgs) { ((tupleArgs.initLAK()), ...); }, binningData.axisTuple);

                //  Do binning for species. Writes to histBuffer
                std::apply(
                    [&](auto const&... tupleArgs) { ((doBinningForSpecies(tupleArgs, currentStep)), ...); },
                    binningData.speciesTuple);

                /**
                 * Deal with time averaging and notify
                 * if period, print output, reset memory to zero
                 */

                /**
                 * Time averaging on n, means accumulate, then average over N notifies
                 * dumpPeriod == 0 is the same as 1. No averaging, output at every step
                 */
                ++accumulateCounter;
                if(accumulateCounter >= binningData.dumpPeriod)
                {
                    auto outputUnits = binningData.depositionData.units;

                    // do the mpi reduce
                    this->histBuffer->deviceToHost();
                    auto bufferExtent = this->histBuffer->getDeviceBuffer().getDataSpace();

                    // allocate this only once?
                    auto hReducedBuffer = HostBuffer<TDepositedQuantity, 1>(bufferExtent);

                    reduce(
                        pmacc::math::operation::Add(),
                        hReducedBuffer.getBasePointer(),
                        this->histBuffer->getHostBuffer().getBasePointer(),
                        bufferExtent[0], // this is a 1D dataspace, just access it?
                        mpi::reduceMethods::Reduce());

                    if(binningData.dumpPeriod > 1 && binningData.timeAveraging)
                    {
                        TDepositedQuantity factor = 1.0 / static_cast<double>(binningData.dumpPeriod);

                        constexpr uint32_t blockSize = 256u;
                        // TODO is + blocksize - 1/ blocksize a better ceil for ints
                        auto gridSize = (bufferExtent[0] + blockSize - 1) / blockSize;
                        // auto gridSize = ceil(
                        //     static_cast<double>(bufferExtent[0]) /
                        //     static_cast<double>(blockSize));

                        auto workerCfg = pmacc::lockstep::makeWorkerCfg<blockSize>();

                        auto productKernel = ProductKernel<blockSize, TBinningData::getNAxes()>();

                        PMACC_LOCKSTEP_KERNEL(productKernel, workerCfg)
                        (gridSize)(binningData.axisExtentsND, factor, hReducedBuffer.getDataBox());
                    }

                    // TODO When doing time averaging, this normalization is not be stable across time for auto bins
                    if(binningData.normalizeByBinVolume)
                    {
                        // TODO think about printing out the normalization too
                        constexpr uint32_t blockSize = 256u;
                        // ceil
                        auto gridSize = (bufferExtent[0] + blockSize - 1) / blockSize;

                        auto workerCfg = pmacc::lockstep::makeWorkerCfg<blockSize>();

                        auto normKernel = BinNormalizationKernel<blockSize, TBinningData::getNAxes()>();

                        auto axisKernels
                            = tupleMap(binningData.axisTuple, [&](auto axis) { return axis.getAxisKernel(); });

                        PMACC_LOCKSTEP_KERNEL(normKernel, workerCfg)
                        (gridSize)(binningData.axisExtentsND, axisKernels, hReducedBuffer.getDataBox());

                        // change output dimensions
                        apply(
                            [&](auto const&... tupleArgs)
                            { ((dimensionSubtraction(outputUnits, tupleArgs.units)), ...); },
                            binningData.axisTuple);
                    }
                    // print output from master
                    if(reduce.hasResult(mpi::reduceMethods::Reduce()))
                    {
                        histWriter(
                            hReducedBuffer,
                            binningData,
                            std::string("binningOpenPMD/"),
                            outputUnits,
                            currentStep);
                    }
                    // reset device buffer
                    this->histBuffer->getDeviceBuffer().setValue(TDepositedQuantity(0.0));
                    accumulateCounter = 0;
                }
            }

            void restart(uint32_t restartStep, const std::string restartDirectory)
            {
                /* restart from a checkpoint here
                 * will be called only once per simulation and before notify() */
            }

            void checkpoint(uint32_t currentStep, const std::string restartDirectory)
            {
                /* create a persistent checkpoint here
                 * will be called before notify() if both will be called for the same timestep */
            }

        private:
            template<typename T_Species>
            void doBinningForSpecies(T_Species, uint32_t currentStep)
            {
                using Species = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_Species>;
                using TParticlesBox = typename Species::ParticlesBoxType;
                using TDataBox = DataBox<PitchedBox<TDepositedQuantity, TBinningData::getNAxes()>>;
                using TDepositedQuantityFunctor = typename TBinningData::DepositionFunctorType;
                // using TDepositedQuantityFunctor = decltype(binningData.quantityFunctor);


                DataConnector& dc = Environment<>::get().DataConnector();
                auto particles = dc.get<Species>(Species::FrameType::getName());

                // TODO do species filtering

                TParticlesBox particlesBox = particles->getDeviceParticlesBox();
                auto binningBox = histBuffer->getDeviceBuffer().getDataBox();

                auto cellDesc = *cellDescription;
                auto const mapper = makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(cellDesc);

                auto globalOffset = Environment<SIMDIM>::get().SubGrid().getGlobalDomain().offset;
                auto localOffset = Environment<SIMDIM>::get().SubGrid().getLocalDomain().offset;

                auto workerCfg = pmacc::lockstep::makeWorkerCfg<Species::FrameType::frameSize>();

                auto axisKernels = tupleMap(binningData.axisTuple, [&](auto axis) { return axis.getAxisKernel(); });

                using TAxisTuple = decltype(axisKernels);
                auto functorBlock = FunctorBlock<
                    TParticlesBox,
                    decltype(binningBox),
                    TDepositedQuantityFunctor,
                    decltype(axisKernels),
                    TBinningData::getNAxes()>(
                    particlesBox,
                    binningBox,
                    binningData.depositionData.functor,
                    axisKernels,
                    globalOffset,
                    localOffset,
                    currentStep,
                    binningData.axisExtentsND);

                PMACC_LOCKSTEP_KERNEL(functorBlock, workerCfg)
                (mapper.getGridDim())(mapper);
            }
        };

    } // namespace plugins::binning
} // namespace picongpu