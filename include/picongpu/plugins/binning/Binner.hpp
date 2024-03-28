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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/plugins/binning/BinningFunctors.hpp"
#include "picongpu/plugins/binning/WriteHist.hpp"
#include "picongpu/plugins/binning/utility.hpp"

#include <pmacc/mpi/MPIReduce.hpp>

#include <cstdint>
#include <memory>
#include <optional>

#include <openPMD/Series.hpp>


namespace picongpu
{
    namespace plugins::binning
    {
        /**
         * Functor to run on divide for time averaging
         */
        // @todo make this generic? apply a functor on all elements of a databox? take in functor?
        template<uint32_t blockSize, uint32_t nAxes>
        struct ProductKernel
        {
            using ResultType = void;

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
                auto blockIdx = worker.blockDomIdxND().x() * blockSize;
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
        // @todo make this more generic? databox operations with another databox?
        // maybe store the normalization in a buffer rather than computing it at every output timestep (memory vs ops)
        // this style vs having the members as params of operator()
        template<uint32_t blockSize, uint32_t nAxes>
        struct BinNormalizationKernel
        {
            using ResultType = void;

            HINLINE
            BinNormalizationKernel()
            {
            }

            // @todo check if type stored in histBox is same as axisKernelTuple Type
            template<typename T_Worker, typename T_DataSpace, typename T_AxisKernelTuple, typename T_DataBox>
            HDINLINE void operator()(
                const T_Worker& worker,
                const T_DataSpace& extentsDataspace,
                const T_AxisKernelTuple& axisKernelTuple,
                T_DataBox histBox) const
            {
                // @todo check normDataBox shape is same as histBox
                auto blockIdx = worker.blockDomIdxND().x() * blockSize;
                auto forEachElemInDataboxChunk = lockstep::makeForEach<blockSize>(worker);
                forEachElemInDataboxChunk(
                    [&](int32_t const linearIdx)
                    {
                        int32_t const linearTid = blockIdx + linearIdx;
                        if(linearTid < extentsDataspace.productOfComponents())
                        {
                            pmacc::DataSpace<nAxes> const nDIdx = pmacc::math::mapToND(extentsDataspace, linearTid);
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

        HINLINE void dimensionSubtraction(std::array<double, 7>& outputDims, const std::array<double, 7>& axisDims)
        {
            for(size_t i = 0; i < 7; i++)
            {
                outputDims[i] += -axisDims[i];
            }
        }

        template<typename TBinningData>
        class Binner : public IPlugin
        {
            using TDepositedQuantity = typename TBinningData::DepositedQuantityType;

            friend class BinningCreator;

        private:
            std::string pluginName; /** @brief name used for restarts */
            TBinningData binningData;
            MappingDesc* cellDescription;
            std::unique_ptr<HostDeviceBuffer<TDepositedQuantity, 1>> histBuffer;
            uint32_t accumulateCounter = 0;
            mpi::MPIReduce reduce{};
            bool isMain = false;
            WriteHist histWriter;
            std::optional<::openPMD::Series> m_series;

        public:
            Binner(TBinningData bd, MappingDesc* cellDesc) : binningData{bd}, cellDescription{cellDesc}
            {
                this->pluginName = "binner_" + binningData.binnerOutputName;
                /**
                 * Allocate and manage global histogram memory here, to facilitate time averaging
                 * @todo for auto n_bins. allocate full size buffer here. dont init axisExtents yet
                 */
                this->histBuffer = std::make_unique<HostDeviceBuffer<TDepositedQuantity, 1>>(
                    binningData.axisExtentsND.productOfComponents());
                isMain = reduce.hasResult();
            }

            ~Binner() override
            {
                if(m_series.has_value())
                {
                    m_series->close();
                }
            }

            void notify(uint32_t currentStep) override
            {
                // @todo auto range init. Init ranges and AxisKernels
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
                    auto bufferExtent = this->histBuffer->getHostBuffer().capacityND();

                    // Do time Averaging
                    if(binningData.dumpPeriod > 1 && binningData.timeAveraging)
                    {
                        TDepositedQuantity factor = 1.0 / static_cast<double>(binningData.dumpPeriod);

                        constexpr uint32_t blockSize = 256u;
                        // @todo is + blocksize - 1/ blocksize a better ceil for ints
                        auto gridSize = (bufferExtent[0] + blockSize - 1) / blockSize;

                        auto productKernel = ProductKernel<blockSize, TBinningData::getNAxes()>();

                        PMACC_LOCKSTEP_KERNEL(productKernel)
                            .template config<blockSize>(gridSize)(
                                binningData.axisExtentsND,
                                factor,
                                this->histBuffer->getDeviceBuffer().getDataBox());
                    }

                    // A copy in case units change during normalization, and we need the original units for OpenPMD
                    auto outputUnits = binningData.depositionData.units;
                    // @todo During time averaging, this normalization is not be constant across time for auto bins
                    // Do normalization
                    if(binningData.normalizeByBinVolume)
                    {
                        // @todo think about printing out the normalization too
                        constexpr uint32_t blockSize = 256u;
                        // ceil
                        auto gridSize = (bufferExtent[0] + blockSize - 1) / blockSize;

                        auto normKernel = BinNormalizationKernel<blockSize, TBinningData::getNAxes()>();

                        auto axisKernels
                            = tupleMap(binningData.axisTuple, [&](auto axis) { return axis.getAxisKernel(); });

                        PMACC_LOCKSTEP_KERNEL(normKernel)
                            .template config<blockSize>(gridSize)(
                                binningData.axisExtentsND,
                                axisKernels,
                                this->histBuffer->getDeviceBuffer().getDataBox());

                        // change output dimensions
                        apply(
                            [&](auto const&... tupleArgs)
                            { ((dimensionSubtraction(outputUnits, tupleArgs.units)), ...); },
                            binningData.axisTuple);
                    }


                    // do the mpi reduce
                    this->histBuffer->deviceToHost();

                    // allocate this only once?
                    // using a unique_ptr here since HostBuffer does not implement move semantics
                    auto hReducedBuffer = std::make_unique<HostBuffer<TDepositedQuantity, 1>>(bufferExtent);

                    reduce(
                        pmacc::math::operation::Add(),
                        hReducedBuffer->data(),
                        this->histBuffer->getHostBuffer().data(),
                        bufferExtent[0],
                        mpi::reduceMethods::Reduce());

                    if(isMain)
                    {
                        // print output from master
                        histWriter(
                            m_series,
                            OpenPMDWriteParams{
                                std::string("binningOpenPMD/"),
                                binningData.binnerOutputName,
                                binningData.openPMDInfix,
                                binningData.openPMDExtension,
                                binningData.jsonCfg},
                            std::move(hReducedBuffer),
                            binningData,
                            outputUnits,
                            currentStep);
                    }
                    // reset device buffer
                    this->histBuffer->getDeviceBuffer().setValue(TDepositedQuantity(0.0));
                    accumulateCounter = 0;
                }
            }

            void pluginRegisterHelp(po::options_description& desc) override
            {
            }

            std::string pluginGetName() const override
            {
                return pluginName;
            }

            void checkpoint(uint32_t currentStep, const std::string restartDirectory) override
            {
                /**
                 * State to hold, accumulateCounter and hReducedBuffer
                 */

                // do the mpi reduce (can be avoided if the notify did a data dump and histBuffer is empty)
                this->histBuffer->deviceToHost();
                auto bufferExtent = this->histBuffer->getHostBuffer().capacityND();

                // allocate this only once?
                // using a unique_ptr here since HostBuffer does not implement move semantics
                auto hReducedBuffer = std::make_unique<HostBuffer<TDepositedQuantity, 1>>(bufferExtent);

                reduce(
                    pmacc::math::operation::Add(),
                    hReducedBuffer->data(),
                    this->histBuffer->getHostBuffer().data(),
                    bufferExtent[0], // this is a 1D dataspace, just access it?
                    mpi::reduceMethods::Reduce());

                if(isMain)
                {
                    std::optional<::openPMD::Series> ckpt_series;

                    histWriter(
                        ckpt_series,
                        OpenPMDWriteParams{
                            restartDirectory + std::string("/binningOpenPMD/"),
                            binningData.binnerOutputName,
                            binningData.openPMDInfix,
                            binningData.openPMDExtension,
                            binningData.jsonCfg},
                        std::move(hReducedBuffer),
                        binningData,
                        binningData.depositionData.units,
                        currentStep,
                        true,
                        accumulateCounter);
                }
            }

            void restart(uint32_t restartStep, const std::string restartDirectory) override
            {
                // retore to master or restore equal values to all MPI ranks or restore only on dump,
                // bool wasRestarted, and read from file and add to buffer

                if(isMain)
                {
                    // open file
                    auto const& extension = binningData.openPMDExtension;
                    std::ostringstream filename;
                    filename << restartDirectory << "/binningOpenPMD/" << binningData.binnerOutputName;
                    if(auto& infix = binningData.openPMDInfix; !infix.empty())
                    {
                        if(*infix.begin() != '_')
                        {
                            filename << '_';
                        }
                        if(*infix.rbegin() == '.')
                        {
                            filename << infix.substr(0, infix.size() - 1);
                        }
                        else
                        {
                            filename << infix;
                        }
                    }
                    if(*extension.begin() == '.')
                    {
                        filename << extension;
                    }
                    else
                    {
                        filename << '.' << extension;
                    }

                    auto openPMDdataFile = ::openPMD::Series(filename.str(), ::openPMD::Access::READ_ONLY);
                    // restore accumulate counter
                    accumulateCounter
                        = openPMDdataFile.iterations[restartStep].getAttribute("accCounter").get<uint32_t>();
                    // restore hostBuffer
                    ::openPMD::MeshRecordComponent dataset
                        = openPMDdataFile.iterations[restartStep]
                              .meshes["Binning"][::openPMD::RecordComponent::SCALAR];
                    ::openPMD::Extent extent = dataset.getExtent();
                    ::openPMD::Offset offset(extent.size(), 0);
                    dataset.loadChunk(
                        std::shared_ptr<float_X>{histBuffer->getHostBuffer().data(), [](auto const*) {}},
                        offset,
                        extent);
                    openPMDdataFile.flush();
                    openPMDdataFile.iterations[restartStep].close();

                    // @todo divide histBuffer by gc.getGlobalSize and call from all ranks

                    // transfer restored data to device so that it is not overwritten
                    this->histBuffer->hostToDevice();
                }
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

                // @todo do species filtering

                TParticlesBox particlesBox = particles->getDeviceParticlesBox();
                auto binningBox = histBuffer->getDeviceBuffer().getDataBox();

                auto cellDesc = *cellDescription;
                auto const mapper = makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(cellDesc);

                auto globalOffset = Environment<SIMDIM>::get().SubGrid().getGlobalDomain().offset;
                auto localOffset = Environment<SIMDIM>::get().SubGrid().getLocalDomain().offset;

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

                PMACC_LOCKSTEP_KERNEL(functorBlock).config(mapper.getGridDim(), particlesBox)(mapper);
            }

            void pluginLoad() override
            {
                Environment<>::get().PluginConnector().setNotificationPeriod(this, binningData.notifyPeriod);
            }
        };

    } // namespace plugins::binning
} // namespace picongpu
