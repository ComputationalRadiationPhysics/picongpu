/* Copyright 2023-2024 Tapish Narwal
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

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/binning/UnitConversion.hpp"
#    include "picongpu/plugins/binning/WriteHist.hpp"

#    include <pmacc/math/operation/traits.hpp>
#    include <pmacc/meta/errorHandlerPolicies/ReturnType.hpp>
#    include <pmacc/mpi/MPIReduce.hpp>
#    include <pmacc/mpi/reduceMethods/Reduce.hpp>

#    include <cstdint>
#    include <memory>
#    include <optional>

#    include <mpi.h>
#    include <openPMD/Series.hpp>

namespace picongpu
{
    namespace plugins::binning
    {
        HINLINE void dimensionSubtraction(
            std::array<double, numUnits>& outputDims,
            std::array<double, numUnits> const& axisDims)
        {
            for(size_t i = 0; i < 7; i++)
            {
                outputDims[i] += -axisDims[i];
            }
        }

        template<typename TBinningData>
        class Binner : public IPlugin
        {
        public:
            using TDepositedQuantity = typename TBinningData::DepositedQuantityType;
            using ReductionOp = typename TBinningData::ReductionOp;
            friend class BinningCreator;

        protected:
            std::string pluginName; /** @brief name used for restarts */
            TBinningData binningData;
            MappingDesc* cellDescription;
            std::unique_ptr<HostDeviceBuffer<TDepositedQuantity, 1>> histBuffer;
            uint32_t reduceCounter = 0;
            mpi::MPIReduce reduce{};
            bool isMain = false;
            WriteHist histWriter;
            std::optional<::openPMD::Series> m_series;

        public:
            Binner(TBinningData const& bd, MappingDesc* cellDesc) : binningData{bd}, cellDescription{cellDesc}
            {
                this->pluginName = "binner_" + binningData.binnerOutputName;
                /**
                 * Allocate and manage global histogram memory here, to facilitate time averaging
                 * @todo for auto n_bins. allocate full size buffer here. dont init axisExtents yet
                 */
                auto const nBins = binningData.axisExtentsND.productOfComponents();
                auto const sizeMiB = nBins * sizeof(TDepositedQuantity) / (1024u * 1024u);
                try
                {
                    this->histBuffer = std::make_unique<HostDeviceBuffer<TDepositedQuantity, 1>>(nBins);
                }
                catch(std::exception const& e)
                {
                    throw std::runtime_error(
                        "Binner '" + this->pluginName + "': failed to allocate histogram buffer ("
                        + std::to_string(nBins) + " bins, " + std::to_string(sizeMiB) + " MiB). "
                        + "Check your axis bin counts. Original error: " + e.what());
                }
                this->histBuffer->getDeviceBuffer().setValue(
                    pmacc::math::operation::traits::NeutralElement<ReductionOp, TDepositedQuantity>::value);
                isMain = reduce.hasResult(mpi::reduceMethods::Reduce());
            }

            ~Binner() override
            {
                if(m_series.has_value())
                {
                    try
                    {
                        m_series->close();
                    }
                    catch(std::exception const& e)
                    {
                        std::cerr << "Binner '" << pluginName << "': error closing openPMD series: " << e.what()
                                  << std::endl;
                    }
                    catch(...)
                    {
                        std::cerr << "Binner '" << pluginName << "': unknown error closing openPMD series"
                                  << std::endl;
                    }
                }
            }

            void notify(uint32_t currentStep) override
            {
                // @todo auto range init. Init ranges and AxisKernels
                doBinning(currentStep);

                ++reduceCounter;
                if(reduceCounter >= binningData.dumpPeriod)
                {
                    auto hReducedBuffer = getReducedBuffer();

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
                                binningData.openPMDJsonCfg},
                            std::move(hReducedBuffer),
                            binningData,
                            currentStep,
                            reduceCounter);
                    }
                    // reset device buffer
                    this->histBuffer->getDeviceBuffer().setValue(
                        pmacc::math::operation::traits::NeutralElement<ReductionOp, TDepositedQuantity>::value);
                    reduceCounter = 0;
                }
            }

            void pluginRegisterHelp(po::options_description& desc) override
            {
            }

            std::string pluginGetName() const override
            {
                return pluginName;
            }

            // if reduceCounter is zero, i.e. notify did a data dump and histBuffer is empty, we still create a
            // checkpoint with an empty buffer, so that restart doesnt throw a warning about missing a checkpoint
            void checkpoint(uint32_t currentStep, std::string const restartDirectory) override
            {
                /**
                 * State to hold, reduceCounter and hReducedBuffer
                 */
                auto hReducedBuffer = getReducedBuffer();

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
                            binningData.openPMDJsonCfg},
                        std::move(hReducedBuffer),
                        binningData,
                        currentStep,
                        reduceCounter);
                }
            }

            void restart(uint32_t restartStep, std::string const restartDirectory) override
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

                    try
                    {
                        auto openPMDdataFile = ::openPMD::Series(filename.str(), ::openPMD::Access::READ_ONLY);
                        // restore reduction counter
                        reduceCounter
                            = openPMDdataFile.iterations[restartStep].getAttribute("reduceCounter").get<uint32_t>();
                        // restore hostBuffer
                        ::openPMD::MeshRecordComponent dataset
                            = openPMDdataFile.iterations[restartStep]
                                  .meshes["Binning"][::openPMD::RecordComponent::SCALAR];
                        ::openPMD::Extent extent = dataset.getExtent();
                        ::openPMD::Offset offset(extent.size(), 0);
                        dataset.loadChunk(
                            std::shared_ptr<TDepositedQuantity>{
                                histBuffer->getHostBuffer().data(),
                                [](auto const*) {}},
                            offset,
                            extent);
                        openPMDdataFile.flush();
                        openPMDdataFile.iterations[restartStep].close();

                        // @todo divide histBuffer by gc.getGlobalSize and call from all ranks

                        // transfer restored data to device so that it is not overwritten
                        this->histBuffer->hostToDevice();
                    }
                    catch(...)
                    {
                        std::cout << "Warning! Unable to load binning data from checkpoint. Will start a new binning, "
                                     "with reduce counter starting from 0 at the first notify after restart."
                                  << std::endl;
                    }
                }

                // Broadcast the reduce counter from rank 0 to all other ranks
                auto& gc = Environment<simDim>::get().GridController();
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Bcast(&reduceCounter, 1, MPI_UINT32_T, 0, gc.getCommunicator().getMPIComm()));
            }

        private:
            void pluginLoad() override
            {
                Environment<>::get().PluginConnector().setNotificationPeriod(this, binningData.notifyPeriod);
            }

            void pluginUnload() override
            {
                if(!binningData.notifyPeriod.empty() && binningData.dumpPeriod > 1 && reduceCounter != 0)
                {
                    auto hReducedBuffer = getReducedBuffer();

                    if(isMain)
                    {
                        std::optional<::openPMD::Series> unload_series;

                        histWriter(
                            unload_series,
                            OpenPMDWriteParams{
                                std::string("binningOpenPMD/"),
                                std::string("end_of_run_") + binningData.binnerOutputName,
                                binningData.openPMDInfix,
                                binningData.openPMDExtension,
                                binningData.openPMDJsonCfg},
                            std::move(hReducedBuffer),
                            binningData,
                            Environment<>::get().SimulationDescription().getRunSteps() - 1,
                            reduceCounter);
                    }
                }
            }

            virtual void doBinning(uint32_t currentStep) = 0;

            std::unique_ptr<HostBuffer<TDepositedQuantity, 1>> getReducedBuffer()
            {
                // do the mpi reduce
                this->histBuffer->deviceToHost();
                auto bufferExtent = this->histBuffer->getHostBuffer().capacityND();

                // Reduces memory footprint compared to allocating this only once and keeping it around
                // ms range cost on every dump
                // using a unique_ptr here since HostBuffer does not implement move semantics
                auto hReducedBuffer = std::make_unique<HostBuffer<TDepositedQuantity, 1>>(bufferExtent);

                reduce(
                    ReductionOp(),
                    hReducedBuffer->data(),
                    this->histBuffer->getHostBuffer().data(),
                    bufferExtent[0], // this is a 1D dataspace, just access it?
                    mpi::reduceMethods::Reduce());
                return hReducedBuffer;
            }
        };


    } // namespace plugins::binning
} // namespace picongpu

#endif
